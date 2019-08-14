from merlin.modules.activation import Activation, NormalizingActivation
from merlin.modules.normalization import Normalization

from alchemy.networks.xception import Xception, Xception65
from alchemy.models.deeplab.model import (Decoder,
                                          Encoder,
                                          Predictor,
                                          Pyramid)


class DeepLab(Predictor.Config):
    """
    The DeepLab v3+ configuration that matches Google's reference
    implementation using Xception 65.
    """

    ACTIVATION = NormalizingActivation.Config(
        activation=Activation.Config(
            kind='relu'
        ),
        normalization=Normalization.Config(
            kind='batch_normalization',
            # This matches the epsilon used in Google's reference DeepLab implementation.
            # Using a different epislon can result in significant deviations in low variance channels.
            epsilon=1e-5
        ),
        # Follows the reference in placing the batch norm before the relu
        activate_before_normalize=False,
        # The batch norm absorbs the bias term
        absorbs_bias=True
    )

    def __init__(self, output_stride=8, logit_channels=None):
        super().__init__(
            encoder=Encoder.Config(
                secondary=[
                    'entry_flow_2/main_branch/conv_1/pointwise/activation/batch_normalization'
                ],
                feature_extractor=Xception(**Xception65(output_stride=output_stride))
            ),

            decoder=Decoder.Config(
                output_stride=4,
                projection_depth=48,
                decoder_depth=256,
                num_refining_units=2,
                num_secondary_feature_maps=1,
                activation=self.ACTIVATION,
            ),

            pyramid=Pyramid.Config(
                output_channels=256,
                dilation_rates=(12, 24, 36),
                activation=self.ACTIVATION,
            ),

            logit_channels=logit_channels
        )
