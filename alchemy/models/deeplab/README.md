# DeepLab v3+

An implementation of the semantic segmentation model described [in this paper](https://arxiv.org/abs/1802.02611):

    Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
    ECCV 2018

-   Based on the [reference implementation](https://github.com/tensorflow/models/tree/master/research/deeplab) by Google.
-   Currently only includes code for inference.

## Usage

See [segment.py](segment.py) for an exampe on how to use the model. The included demo app can be invoked from the command line as follows:

```
python alchemy/models/deeplab/segment.py --input [input image path] --snapshot [snapshot directory path] --output /tmp/output.png
```

-   Replace `[input image path]` with any valid `jpeg` or `png` path.
-   Use any of the extracted models below for `[snapshot directory path]`.

## Pre-trained models

The model weights below were ported from the ones published by Google.

-   [DeepLab v3+ with Xception65 trained on PASCAL VOC](https://dl.bintray.com/ethereon/alchemy/alchemy-deeplab-xception65-pascal.tgz)

---

![Sample output using this implementation](https://dl.bintray.com/ethereon/alchemy/sample-output.jpg)
