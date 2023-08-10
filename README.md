### Food vision project

Build a classification model on the [food101 dataset](https://www.tensorflow.org/datasets/catalog/food101) using tensorflow

This dataset consists of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.

The model employs ``mixed-precision`` training within the ``TensorFlow framework``, utilizing `transfer learning` techniques that encompass both `feature extraction` and `fine-tuning` stages. This approach is executed on the `EfficientNetB0` architecture, which serves as the foundational backbone for the neural network.