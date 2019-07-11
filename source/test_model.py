from load_data import load_test_data

from tensorflow.python import keras
from tensorflow.python.keras.models import model_from_json

# Set hyper parameters
learning_rate = 0.03
momentum = 0.9

# Load JSON file and build model
json_file = open("./../model/model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load weights into model
model.load_weights("./../model/model.h5")

# Set the loss function and optimizer
model.compile(loss = keras.losses.categorical_crossentropy, 
	optimizer = keras.optimizers.SGD(lr = learning_rate, momentum = momentum), metrics = ["accuracy"])

# Load the test data
test_images, test_labels = load_test_data()

# Evaluate the model on the test set
score = model.evaluate(test_images, test_labels, verbose=0)
print("Accuracy: %1.3f%%" % (score[1]*100))