"""
Command line interface for an ImageNet classifier.
"""

import argparse

import tensorflow as tf

from merlin.ops.image import resample, load_image
from merlin.context import inferential
from merlin.snapshot import load_snapshot

from alchemy.models.classifier.model import ImageClassifier
from alchemy.models.classifier.presets import xception_65_classifier
from alchemy.models.classifier.labels import imagenet


def preprocess(image, size=(256, 256)):
    image = tf.cast(image, tf.float32)
    image = (image * 2. / 255.) - 1.0
    image = image[tf.newaxis, ...]
    image = resample(image, size=size, method=resample.BILINEAR)
    return image


@inferential
def classify_top_k(image, classifier, k):
    # Pre-process and classify
    logits = classifier(preprocess(image))

    # Get the top k classes
    _, top_k_indices = tf.math.top_k(tf.squeeze(logits), k=k)

    # Display results
    labels = ('Background',) + imagenet.LABELS
    for rank, class_idx in enumerate(top_k_indices):
        print(f'{rank + 1} : {labels[class_idx]}')


def main():
    # Parse args
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        '--image',
        required=True,
        help='Path to an image file.'
    )
    parser.add_argument(
        '--snapshot',
        required=True,
        help='Path to a classification model checkpoint directory.'
    )
    args = parser.parse_args()

    # Load the image
    image = load_image(args.image)
    # Create the classifier
    classifier = ImageClassifier(**xception_65_classifier(num_classes=1001))
    # Load the weights
    load_snapshot(directory=args.snapshot, model=classifier)

    # Display the top-5 most likely classifications
    classify_top_k(image=image, classifier=classifier, k=5)


if __name__ == "__main__":
    main()
