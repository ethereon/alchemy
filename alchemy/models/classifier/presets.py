from alchemy.models.classifier.model import ImageClassifier


def xception_65_classifier(num_classes):
    from alchemy.networks.xception import Xception, Xception65

    return ImageClassifier.Config(
        feature_extractor=Xception(**Xception65()),
        num_classes=num_classes
    )
