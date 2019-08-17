# Xception

This is a variant of the original Xception network:

    Xception: Deep Learning with Depthwise Separable Convolutions
    https://arxiv.org/abs/1610.02357

as specified in the DeepLab v3+ work:

    Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
    https://arxiv.org/abs/1802.02611

which attributes some of the modifications to:

    Deformable Convolutional Networks
    COCO Detection and Segmentation Challenge 2017 Entry
    http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf

Google's reference implementation mentions the following modifications:

-   Support for aligned feature maps
-   Fully Convolutional: max pooling layers replaced with separable convolutions with
    `stride = 2`
-   Dilated Convolutions: used for achieving arbitrary output strides (see [presets.py](presets.py))
-   ReLU and Batch Normalization after depthwise convolution (motivated by MobileNet v1)

## Pre-trained models

-   [Semantic Segmentation](../../models/deeplab)
-   [Image Classification](../../models/classifier)
