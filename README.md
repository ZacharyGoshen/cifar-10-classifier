# Image Classification Using a Convolutional Neural Network

The CIFAR-10 data set is a labeled subset of the 80 million tiny images data set. CIFAR-10 contains images of ten different classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The data set is split into a training set of 50,000 images (5,000 per class) and a test set of 10,000 images (1000 per class). Each image is a 32 by 32 RGB image of one of the ten classes.

*train_model.py* creates and trains a convolutional neural network to classify the test images. The CNN is based off of the VGG-B network outlined in https://arxiv.org/pdf/1409.1556.pdf. VGG is used to classify very large images (224 by 224) so some modifications were made to better fit the smaller data of CIFAR-10.
