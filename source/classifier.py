import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential

def load_batch(file_num):
	# Construct specified file path
	file_name = "./../data/"
	if (file_num != 6):
		file_name += "data_batch_" + str(file_num)
	else:
		file_name += "test_batch"

	# Open file and retrieve data
	data = pickle.load(open(file_name, "rb"), encoding = "bytes")

	# Format the image data to dimensions (?, 32, 32, 3)
	images = data[b'data']
	images = np.reshape(images, (10000, 3, 32, 32))
	images = np.swapaxes(images, 1, 3)
	images = np.swapaxes(images, 1, 2)

	# Rescale the pixel to [0, 1]
	images = np.divide(images, 255)

	# Format the labels as one hot encodings with 10 classes
	labels_temp = data[b'labels']
	labels = np.zeros((10000, 10))
	for i in range(0, 10000):
		labels[i, labels_temp[i]] = 1

	return (images, labels)

# Load the first batch of training data
training_images, training_labels = load_batch(1) 

# Load and append the remaining training batches to the training data
for batch in range(2, 6):
	batch_images, batch_labels = load_batch(batch)
	training_images = np.append(training_images, batch_images, axis = 0)
	training_labels = np.append(training_labels, batch_labels, axis = 0)

# Load the test data
test_images, test_labels = load_batch(6) 

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

class AccuracyHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs = {}):
		self.acc = []

	def on_epoch_end(self, batch, logs = {}):
		self.acc.append(logs.get('acc'))

# Create a callback to record the accuracy after each epoch
history = AccuracyHistory()

# Set the loss function and optimizer
model.compile(loss = keras.losses.categorical_crossentropy, 
	optimizer = keras.optimizers.SGD(lr = learning_rate, momentum = momentum), metrics = ["accuracy"])

# Train the models
model.fit(training_images, training_labels, batch_size = batch_size, epochs = num_epochs,
 verbose = 1, validation_data = [test_images, test_labels], callbacks = [history])

# Save the weights for the model
model.save_weights("./../model/model.h5")

# Plot the accuracy over the course of training
plt.plot(range(1, num_epochs + 1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()