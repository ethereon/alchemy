from typing import Sequence, Optional

import tensorflow as tf

from merlin.interceptor import OutputSelector
from merlin.modules.activation import NormalizingActivation
from merlin.modules.convolution import Conv2D, DepthwiseSeparableConv
from merlin.modules.convolution.padding import SamePadding
from merlin.modules.configurable import Module
from merlin.modules.util import Sequential, get_submodule_by_path
from merlin.ops.image import resample, legacy_aligned_resampler
from merlin.spec import Spec
from merlin.shape import Axis, ImageSize, get_spatial_size
from merlin.typing import Tensor, Size2D


class Encoder(Module):
    """
    Convolutional encoder that outputs multi-level feature maps.
    """

    class Config(Spec):
        # One or more feature extractor sub-module names.
        # These are used for extracting multi-scale feature maps.
        secondary: Sequence[str]
        # A feature extractor instance (such as Xception 65).
        # The instance must be indexable with the secondary feature names.
        feature_extractor: tf.Module
        # Optional name
        name: Optional[str] = None

    def configure(self, config: Config):
        self.feature_extractor = config.feature_extractor
        self.selector = OutputSelector(*(
            get_submodule_by_path(self.feature_extractor, feature)
            for feature in config.secondary
        ))

    def compute(self, inputs: Tensor) -> Sequence[Tensor]:
        with self.selector:
            output = self.feature_extractor(inputs)
        return [output] + self.selector.pop()


class Pyramid(Module):
    """
    Atrous (dilated) Spatial Pyramid Pooling (ASSP)

    Parallel branches of dilated convolutions that extract features
    at multiple spatial scales.
    """

    class Config(Spec):
        # Number of output channels produced by the pyramid.
        output_channels: int
        # The dilation rate for each convolution branch.
        dilation_rates: Sequence[int]
        # An activation + optional normalization function
        activation: NormalizingActivation.Config
        # The kernel size for the dilated convolutions.
        kernel_size: int = 3
        # Whether to use aligned padding.
        use_aligned_padding: bool = True
        # Optional name.
        name: Optional[str] = None

    def configure(self, config: Config):
        self.pooling_projection = Conv2D(
            filters=config.output_channels,
            kernel_size=1,
            use_bias=(not config.activation.absorbs_bias),
            activation=NormalizingActivation(**config.activation),
            name='pooling_projection'
        )

        # Solitary pointwise convolution
        self.branches = [
            Conv2D(
                kernel_size=1,
                filters=config.output_channels,
                use_bias=(not config.activation.absorbs_bias),
                activation=NormalizingActivation(**config.activation),
                name='input_projection'
            )
        ]

        # Dilated depthwise separable convolutions
        self.branches += [
            DepthwiseSeparableConv(
                num_outputs=config.output_channels,
                kernel_size=config.kernel_size,
                depth_multiplier=1,
                strides=1,
                use_bias=(not config.activation.absorbs_bias),
                activation=NormalizingActivation(**config.activation),
                dilation_rate=dilation_rate,
                padding=SamePadding(aligned=config.use_aligned_padding),
                name='dilator'
            )
            for dilation_rate in config.dilation_rates
        ]

        self.output_projection = Conv2D(
            kernel_size=1,
            filters=config.output_channels,
            use_bias=(not config.activation.absorbs_bias),
            activation=NormalizingActivation(**config.activation),
            name='output_projection'
        )

    def compute(self, inputs: Tensor):
        # Globally pool the encoder features, convolve, then upsample.
        # The reference implementation refers to this as "image pooling" /
        # "adding image level features".
        pooled = tf.reduce_mean(
            inputs,
            axis=(Axis.height, Axis.width),
            keepdims=True
        )
        # Convolve the global average pooled (1x1) output
        pooled = self.pooling_projection(pooled)
        # Upsample
        pooled = resample(
            tensor=pooled,
            like=inputs,
            method=resample.NEAREST_NEIGHBOR,
            resampler=legacy_aligned_resampler
        )

        # Compute the output for each parallel branch
        branch_outputs = [branch(inputs) for branch in self.branches]

        # Concatenate everything together
        output = tf.concat([pooled] + branch_outputs, axis=Axis.channel)

        # Apply the final output projection
        return self.output_projection(output)


class Decoder(Module):
    """
    The decoder refines multi-scale feature maps from the encoder to produce
    sharper segmentations.
    """

    class Config(Spec):
        # The decoded output's spatial size is subsampled by this factor
        # relative to the input image's size.
        output_stride: int
        # The number of convolution layers used for refining the segmentation
        num_refining_units: int
        # The number of secondary feature maps that will be provided to the decoder
        num_secondary_feature_maps: int
        # Each secondary feature map is projected to this common number of channels
        projection_depth: int
        # The number of channels in the convolutions used for refining the segmentation
        decoder_depth: int
        # The activation function used for all convolutions in the decoder
        activation: NormalizingActivation.Config
        # The kernel size for the refining convolutions
        decoder_kernel_size: int = 3
        # Optional normali
        # Whether to use aligned "same" padding for the convolutions
        use_aligned_padding: bool = True

    def configure(self, config: Config):
        self.config = config

        # Each secondary feature map is first subjected to a channel projection
        self.secondary_projections = [
            Conv2D(
                filters=config.projection_depth,
                kernel_size=1,
                use_bias=(not config.activation.absorbs_bias),
                activation=NormalizingActivation(**config.activation),
                name='secondary_projection'
            )
            for idx in range(config.num_secondary_feature_maps)
        ]

        self.refiner = Sequential(
            DepthwiseSeparableConv(
                num_outputs=config.decoder_depth,
                kernel_size=config.decoder_kernel_size,
                depth_multiplier=1,
                strides=1,
                dilation_rate=1,
                use_bias=(not config.activation.absorbs_bias),
                activation=NormalizingActivation(**config.activation),
                padding=SamePadding(aligned=config.use_aligned_padding),
                name='refiner'
            )
            for unit in range(config.num_refining_units)
        )

    def compute(
        self,
        primary_features: Tensor,
        secondary_features: Sequence[Tensor],
        input_image_size: Size2D
    ):
        # Project all secondary features to a common number of channels
        assert len(self.secondary_projections) == len(secondary_features)
        projections = [
            proj(feats)
            for proj, feats in zip(self.secondary_projections, secondary_features)
        ]

        # Compute the decoded output size
        decoded_size = input_image_size.scale(
            factor=1 / self.config.output_stride,
            quantize=tf.math.ceil
        )

        # Refine using spatial convolutions
        decoded_features = primary_features
        for projected_secondary_feature in projections:
            # Scale all feature maps to the same size
            decoder_inputs = [
                resample(
                    feature_map,
                    size=decoded_size,
                    method=resample.BILINEAR,
                    resampler=legacy_aligned_resampler
                )
                for feature_map in (decoded_features, projected_secondary_feature)
            ]
            # Stack and convolve
            decoded_features = self.refiner(tf.concat(decoder_inputs, axis=Axis.channel))

        return decoded_features


class Predictor(Module):
    """
    The DeepLab v3+ pixelwise segmentation model as published in:

        Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
        https://arxiv.org/abs/1802.02611

    This class assembles the various individual components of the model into a single module.
    """

    class Config(Spec):
        # Encoder parameters
        encoder: Encoder.Config
        # Decoder parameters
        decoder: Decoder.Config
        # Dilated spatial pyramid pooling parameters
        pyramid: Pyramid.Config

        # An optional number of output logit channels
        # If None, the logit convolution is omitted.
        logit_channels: Optional[int] = None

    def configure(self, config: Config):
        self.encoder = Encoder(config=config.encoder)
        self.decoder = Decoder(config=config.decoder)
        self.pyramid = Pyramid(config=config.pyramid)
        self.logits = (
            Conv2D(
                filters=config.logit_channels,
                kernel_size=1,
                name='logits'
            )
            if config.logit_channels is not None else None
        )

    def compute(self, inputs: Tensor):
        # Encode
        primary_features, *secondary_features = self.encoder(inputs)
        # Dilated spatial pyramid pooling
        pyramid_pooled = self.pyramid(primary_features)
        # Decode
        outputs = self.decoder(
            primary_features=pyramid_pooled,
            secondary_features=secondary_features,
            input_image_size=get_spatial_size(inputs)
        )

        return self.logits(outputs) if self.logits is not None else outputs
