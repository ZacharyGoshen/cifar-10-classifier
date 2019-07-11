from load_data import load_training_data, load_test_data

import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential

# Load the training data
training_images, training_labels = load_training_data()

# Load the test data
test_images, test_labels = load_test_data()

# Set hyper parameters
learning_rate = 0.03
momentum = 0.9
batch_size = 64
num_epochs = 1

# Construct the model
model = Sequential()
model.add(Conv2D(32, (3, 3), strides = (1, 1), padding = "same", activation = "relu", input_shape = (32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), strides = (1, 1), padding = "same", activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = "same", activation = "relu", input_shape = (32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = "same", activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = "same", activation = "relu", input_shape = (32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = "same", activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation = "softmax"))

# Set the loss function and optimizer
model.compile(loss = keras.losses.categorical_crossentropy, 
	optimizer = keras.optimizers.SGD(lr = learning_rate, momentum = momentum), metrics = ["accuracy"])

# Load the weights for the model
model.load_weights("./../model/model.h5")

# Train the models
model.fit(training_images, training_labels, batch_size = batch_size, epochs = num_epochs,
 verbose = 1, validation_data = [test_images, test_labels])

# Save the model as JSON
open("./../model/model.json", "w").write(model.to_json())

# Save the updated weights for the model
model.save_weights("./../model/model.h5")