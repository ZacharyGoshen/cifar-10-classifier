import numpy as np
import pickle

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

def load_training_data():
	# Load the first batch of training data
	training_images, training_labels = load_batch(1) 

	# Load and append the remaining training batches to the training data
	for batch in range(2, 6):
		batch_images, batch_labels = load_batch(batch)
		training_images = np.append(training_images, batch_images, axis = 0)
		training_labels = np.append(training_labels, batch_labels, axis = 0)

	return (training_images, training_labels)

def load_test_data():
	# Load the test data
	test_images, test_labels = load_batch(6) 

	return (test_images, test_labels)