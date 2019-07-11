# Image Classification Using a Convolutional Neural Network

The CIFAR-10 data set is a labeled subset of the 80 million tiny images data set. CIFAR-10 contains images of ten different classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The data set is split into a training set of 50,000 images (5,000 per class) and a test set of 10,000 images (1000 per class). Each image is a 32 by 32 RGB image of one of the ten classes.

*train_model.py* creates and trains a convolutional neural network to classify the test images. The CNN is based off of the VGG-B network outlined in https://arxiv.org/pdf/1409.1556.pdf. VGG is used to classify very large images (224 by 224) so some modifications were made to better fit the smaller data of CIFAR-10. For instance, instead of using five convolutional blocks as in VGG, I used three. And, instead of using 64 filters in the first layer, I used 32.

The model consists of three convolutional blocks and two dense layers. Each convolutional block has two convolutional layers and a max pooling layer. All three blocks use 3 by 3 filters with padding to avoid dimensionality reduction. Each block also uses 2 by 2 max pooling, reducing the dimensions by half. Each block has two convolutional layers followed by a max pooling layer. Both convolutional layers in the first block have output of depth 32. Convolutional layers in the second block have output of depth 64, and 128 in the third block. 

After all the convolution/poolings blocks, the output is flattened and input into the first dense layer with 128 nodes. The final output is run through the second dense layer using softmax to get the ten categorical probabilities. 

Batch normalization is performed after every convolutional layer and the first depth layer to stabilize the learning process. To avoid overfitting the training data, dropout is performed after each convolutional block and the first depth layer. The dropout proportion increases the further you go into the network using values of .1, .2, .3, and .4.




