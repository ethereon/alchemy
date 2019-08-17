import tensorflow as tf

from merlin.modules.configurable import Composite
from merlin.modules.pooling import GlobalAveragePooling
from merlin.modules.convolution import Conv2D
from merlin.spec import Spec


class ImageClassifier(Composite):

    class Config(Spec):
        # The number of classes
        num_classes: int
        # The feature extractor to use
        feature_extractor: tf.Module

    def configure(self, config: Config):
        return (
            config.feature_extractor,
            GlobalAveragePooling(),
            Conv2D(
                filters=config.num_classes,
                kernel_size=1,
                name='logits'
            )
        )
