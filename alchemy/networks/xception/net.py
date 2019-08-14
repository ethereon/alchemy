"""
An implementation of the Xception architecture described in

    Xception: Deep Learning with Depthwise Separable Convolutions
    https://arxiv.org/abs/1610.02357

This version supports dilated convolutions and is compatible with Google's
implementation, as included in the DeepLab project.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Callable

from merlin.modules.convolution.padding import SamePadding
from merlin.modules.convolution import Conv2D, DepthwiseSeparableConv
from merlin.modules.activation import Activation, NormalizingActivation
from merlin.modules.util import Sequential
from merlin.modules.configurable import Composite, Module
from merlin.spec import Spec


class XceptionRoot(Composite):
    """
    The initial group of dense (non-separable) convolutions.
    """

    class Config(Spec):
        # The output channel counts for each convolution
        depths: Iterable[int]
        # The stride for each convolution
        strides: Iterable[int]
        # Wheter to use explicit aligned same padding
        use_aligned_padding: bool
        # An activation + optional normalization
        activation: NormalizingActivation.Config
        # The convolution kernel size
        kernel_size: int = 3
        # Name for the module
        name: Optional[str] = None

    def configure(self, config: Config):
        assert len(config.depths) == len(config.strides)
        return (
            Conv2D(
                kernel_size=config.kernel_size,
                filters=depth,
                strides=stride,
                padding=SamePadding(aligned=config.use_aligned_padding),
                use_bias=(not config.activation.absorbs_bias),
                activation=NormalizingActivation(**config.activation),
                name='root'
            )
            for depth, stride in zip(config.depths, config.strides)
        )


class SkipConnection(str, Enum):
    """
    Various skip conections used by the Xception module.
    """

    # The input is convolved with a 1x1 "projection" kernel
    CONVOLUTION = 'convolution'
    # The input is passed through for summing
    IDENTITY = 'identity'
    # No skip connections
    NONE = 'none'

    def create(self, config: XceptionModule.Config) -> Optional[Callable]:
        """
        Create a callable instance of this skip connection.
        """
        if self == SkipConnection.CONVOLUTION:
            return Conv2D(
                kernel_size=1,
                filters=config.depths[-1],
                strides=config.stride,
                use_bias=(not config.activation.absorbs_bias),
                activation=NormalizingActivation(
                    activation=None,
                    normalization=config.activation.normalization
                ),
                name='shortcut'
            )

        if self == SkipConnection.IDENTITY:
            return lambda x: x

        if self == SkipConnection.NONE:
            return None

        raise ValueError(f'Unsupported skip transform type: {self}')


class XceptionModule(Module):
    """
    A chain of depthwise separable convolutions with an optional skip branch.
    """

    class Config(Spec):
        # These specify the output channel counts for the convolutions in a single
        # module in this group.
        depths: Iterable[int]
        # The convolution stride
        stride: int
        # An activation + optional normalization
        activation: NormalizingActivation.Config
        # The kind of skip connections to use.
        skip_connection: SkipConnection
        # Wheter to use explicit aligned same padding
        use_aligned_padding: bool
        # The convolutional kernel size
        kernel_size: int = 3
        # The dilation factor
        dilation_rate: int = 1
        # The module's name
        name: Optional[str] = None

    def configure(self, config: Config):
        # Determine whether to apply an activation to the input, or to
        # place two of them within the depthwise separable convolutions.
        pre_activate = (config.skip_connection != SkipConnection.NONE)
        conv_activation = (
            config.activation.replace(activation=None)
            if pre_activate else config.activation
        )

        self.main_branch = Sequential(
            (
                Activation(**config.activation.activation) if pre_activate else None,

                DepthwiseSeparableConv(
                    num_outputs=depth,
                    kernel_size=config.kernel_size,
                    depth_multiplier=1,
                    # Only use non-unitary stride for the final convolution
                    strides=(config.stride if idx == (len(config.depths) - 1) else 1),
                    dilation_rate=config.dilation_rate,
                    padding=SamePadding(aligned=config.use_aligned_padding),
                    use_bias=(not config.activation.absorbs_bias),
                    activation=NormalizingActivation(**conv_activation),
                    name='conv'
                )
            )
            for idx, depth in enumerate(config.depths)
        )

        # Skip connection
        self.skip_connection = SkipConnection(config.skip_connection).create(config=config)

    def compute(self, inputs):
        output = self.main_branch(inputs)
        if self.skip_connection:
            output = self.skip_connection(inputs) + output
        return output


class Xception(Composite):
    """
    An xception network consists of the following structure:
        - A series of "root" convolutions
        - A series of "xception modules" (logically grouped in "flows")

    For further details, see the paper:
        Xception: Deep Learning with Depthwise Separable Convolutions
        https://arxiv.org/abs/1610.02357

    This version follows the variant originally implemented in DeepLab v3,
    which adds support for dilated convolutions and controlling the output
    resolution for dense pixelwise predictions.
    """

    class Config(Spec):
        # The root convolutional group config
        root: XceptionRoot.Config
        # The rest of the network is constructed by sequentially stacking
        # a series of xception module instances.
        modules: Iterable[XceptionModule.Config]
        # An optional name for scoping the network
        name: Optional[str] = None

    def configure(self, config: Config):
        return (
            # Root convolutions
            XceptionRoot(**config.root),
            # A series of xception modules
            [XceptionModule(**config) for config in config.modules]
        )
