from typing import Optional

import numpy as np

from merlin.modules.activation import Activation, NormalizingActivation
from merlin.modules.normalization import Normalization
from merlin.util.collections import flatten

from alchemy.networks.xception import Xception, XceptionModule, XceptionRoot, SkipConnection


class XceptionPreset(Xception.Config):
    """
    Base class for Xception network presets.
    """

    def __init__(
        self,
        output_stride: Optional[int] = None,
        activation: NormalizingActivation = None,
        use_aligned_padding: bool = True
    ):
        self._activation = activation or self._get_default_activation()
        self._use_aligned_padding = use_aligned_padding

        super().__init__(**self._setup())

        if output_stride is not None:
            self._reconfigure_for_output_stride(output_stride=output_stride)

    def _get_default_activation(self):
        return NormalizingActivation.Config(
            activation=Activation.Config(kind='relu'),
            normalization=Normalization.Config(kind='batch_normalization'),
            activate_before_normalize=False,
            absorbs_bias=True
        )

    def _module(self, **kwargs):
        return XceptionModule.Config(
            activation=self._activation,
            use_aligned_padding=self._use_aligned_padding,
            **kwargs
        )

    def _reconfigure_for_output_stride(self, output_stride: int):
        """
        Updates the strides and dilation factors for the given network
        to achieve the requested output stride.
        This matches the procedure used in DeepLab v3.

        Throws ValueError if the requested output stride is infeasible.
        """
        if output_stride % 2 != 0:
            raise ValueError('The output stride must be even.')

        # Update the output stride to account for the root convolutions
        output_stride //= np.prod(self.root.strides)

        # Start off with unit stride and dilation.
        current_stride = 1
        current_dilation_rate = 1

        for module in self.modules:
            # Check the output stride feasibility
            if current_stride > output_stride:
                raise ValueError(f'An output stride of {output_stride} is not feasible.')

            # If the requested output stride has been reached, disable strided
            # convolutions and switch to dilation to expand the receptive field.
            if current_stride == output_stride:
                module.dilation_rate = current_dilation_rate
                current_dilation_rate *= module.stride
                module.stride = 1
            else:
                module.dilation_rate = 1
                current_stride *= module.stride

        # Verify if the requested output stride was reached
        if current_stride != output_stride:
            raise ValueError(f'An output stride of {output_stride} is not feasible.')

    def _setup(self):
        raise NotImplementedError


class Xception65(XceptionPreset):
    """
    The "Xception 65" configuration as specified in DeepLab v3+.
    Similar to Google's reference implementation, this version supports
    dilated convolutions.
    """

    def _setup(self):
        return Xception.Config(
            name='xception_65',

            root=XceptionRoot.Config(
                name='entry_flow_root',
                depths=(32, 64),
                strides=(2, 1),
                activation=self._activation,
                use_aligned_padding=self._use_aligned_padding,
            ),

            modules=flatten(
                # Entry flows
                self._module(
                    name='entry_flow_1',
                    depths=(128, 128, 128),
                    skip_connection=SkipConnection.CONVOLUTION,
                    stride=2
                ),
                self._module(
                    name='entry_flow_2',
                    depths=(256, 256, 256),
                    skip_connection=SkipConnection.CONVOLUTION,
                    stride=2
                ),
                self._module(
                    name='entry_flow_3',
                    depths=(728, 728, 728),
                    skip_connection=SkipConnection.CONVOLUTION,
                    stride=2
                ),

                # Middle flows
                [
                    self._module(
                        name='middle_flow',
                        depths=(728, 728, 728),
                        skip_connection=SkipConnection.IDENTITY,
                        stride=1
                    )
                    for _ in range(16)
                ],

                # Exit flows
                self._module(
                    name='exit_flow_1',
                    depths=(728, 1024, 1024),
                    skip_connection=SkipConnection.CONVOLUTION,
                    stride=2
                ),
                self._module(
                    name='exit_flow_2',
                    depths=(1536, 1536, 2048),
                    skip_connection=SkipConnection.NONE,
                    stride=1
                ),
            )
        )
