"""
An image segmentation frontend built on top of DeepLab v3.
"""

from typing import Optional

import tensorflow as tf

from merlin.typing import Tensor, Size2D
from merlin.ops.image import load_image, write_image, resample
from merlin.spec import Spec, Default
from merlin.shape import Axis
from merlin.snapshot import load_snapshot
from merlin.util.configurable import Configurable
from merlin.visualize.color import DiscreteColorMapper
from merlin.context import inferential

from alchemy.models.deeplab import DeepLab, Predictor


class ImageSegmenter(Configurable):
    """
    A configurable image segmenter powered by DeepLab v3+.
    """

    class Config(Spec):
        # The predictor model configuration
        predictor: Predictor.Config = Default(DeepLab(
            # By default, the number of channels in the PASCAL dataset
            logit_channels=21
        ))

        # The input resolution expected by the predictor
        # Images that aren't this size will be automatically resized
        # using bilinear interpolation.
        input_size: Size2D = Default((513, 513))

        # Optional path to a model checkpoint to load
        checkpoint_dir: Optional[str] = None

    def configure(self, config: Config):
        self.config = config
        self.predictor = Predictor(config=config.predictor)
        if config.checkpoint_dir:
            load_snapshot(
                directory=config.checkpoint_dir,
                model=self.predictor
            )

    def pre_process(self, image: Tensor) -> Tensor:
        """
        Pre-process an input image before feeding it to the network.
        """
        # Convert to float
        image = tf.cast(image, tf.float32)

        # Normalize from [0, 255] to [-1, 1]
        image = (image * 2. / 255.) - 1.0

        # Ensure rank 4 tensor
        if image.shape.rank == 3:
            # Inject batch dimension
            image = image[tf.newaxis, ...]
        elif image.shape.rank != 4:
            raise ValueError('Input image must be either rank 3 or 4.')

        # Resize to expected input size
        if self.config.input_size:
            image = resample(image, size=self.config.input_size, method=resample.BILINEAR)

        return image

    @inferential
    def segment(self, image: Tensor, match_size=True) -> Tensor:
        """
        Semantically segment the given image and return a single channel image
        where pixel values correspond to class indices.

        If match_size is True, the output is resized to match the input dimensions.
        Otherwise, the output is returned at the original sub-subsampled size.
        """
        # Get the per-class label logits
        logits = self.predictor(self.pre_process(image))

        # Reduce to per-pixel labels corresponding to the most likely class
        labels = tf.argmax(logits, axis=Axis.channel)

        # Resize the label map to the input size if requested
        if match_size:
            labels = tf.squeeze(
                resample(
                    labels[..., tf.newaxis],
                    like=image,
                    method=resample.NEAREST_NEIGHBOR
                )
            )

        return labels


def main():
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        '--input',
        required=True,
        help='Path to an image to segment.'
    )
    parser.add_argument(
        '--snapshot',
        required=True,
        help='Path to a trained DeepLab v3+ checkpoint directory.'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to an image file where the segmentation will be written.'
    )
    args = parser.parse_args()

    # Load the input image
    image = load_image(args.input)
    # Create the segmenter
    segmenter = ImageSegmenter(checkpoint_dir=args.snapshot)
    # Segment the image
    labels = segmenter.segment(image)
    # Visualize the labels by applying a color map
    mapper = DiscreteColorMapper(
        input_cardinality=segmenter.config.predictor.logit_channels,
        zero_color=(0, 0, 0)
    )
    color_labels = mapper(labels)
    # Write the output alongside the original input
    write_image(args.output, tf.concat([image, color_labels], axis=1))


if __name__ == "__main__":
    main()
