#
# This is a sample Notebook(that has been adapted to a python file) to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt

from mlp import MultilayerPerceptron
from mlp import Layer

from ActivationsFunctions.Linear import Linear
from ActivationsFunctions.Mish import Mish
from ActivationsFunctions.Relu import Relu
from ActivationsFunctions.Sigmoid import Sigmoid
from ActivationsFunctions.Softmax import Softmax
from ActivationsFunctions.Softplus import Softplus
from ActivationsFunctions.Tanh import Tanh

from Weights.GlorotUniform import GlorotUniform

from LossFunctions.SquaredError import SquaredError
from LossFunctions.CrossEntropy import CrossEntropy

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        print("Loading Images")
        x_prep, y_prep = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test_raw, y_test_raw = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_prep, y_prep),(x_test_raw, y_test_raw)

#
# Set file paths based on added MNIST Datasets
#
input_path = './mnist/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    print("Showing Images")
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);
        index += 1
    plt.show()

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_prep, y_prep), (x_test_raw, y_test_raw) = mnist_dataloader.load_data()

#
# Show some random training and test images
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_prep[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_prep[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test_raw[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test_raw[r]))

# show_images(images_2_show, titles_2_show)


# Here below is my own code minus the matplotlib code
print("Converting to numpy")
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

training_split = .8
num_split = int(.8*len(x_prep))
x_train, y_train = x_prep[:num_split], y_prep[:num_split]
x_val, y_val = x_prep[num_split:], y_prep[num_split:]
x_test, y_test = x_test_raw, y_test_raw

for i in range(len(x_train)):
    x_train[i] = np.array(x_train[i], dtype=int).flatten()
    y_train[i] = np.array(y_train[i])

for i in range(len(x_val)):
    x_val[i] = np.array(x_val[i], dtype=int).flatten()
    y_val[i] = np.array(y_val[i])

for i in range(len(x_test)):
    x_test[i] = np.array(x_test[i], dtype=int).flatten()
    y_test[i] = np.array(y_test[i])


# I cant for the life of me figure out how to standardize the pixels to not overflow my activation functions
# Granted, Softmax hasn't been completed yet
# It seems like the digits of the gradient are exploding!
x_train = np.vectorize(lambda element: 2 if element > 50 else 1)(x_train)
y_train = np.array(y_train, dtype=int)

x_val = np.vectorize(lambda element: 2 if element > 50 else 1)(x_val)
y_val = np.array(y_val, dtype=int)

x_test = np.vectorize(lambda element: 2 if element > 50 else 1)(x_test)
y_test = np.array(y_test, dtype=int)

print("Making Layers")
layer_window = []

# Need to account for vectors
num_inputs = x_train.shape[1] if len(x_train.shape) > 1 else 1
num_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1

layer_window.append(num_inputs)

# Layer sizes I get to choose
layer_window.append(10)
layer_window.append(15)
layer_window.append(20)
layer_window.append(20)
layer_window.append(10)

layer_window.append(num_outputs)

layers = []

activation_function_bulk = Sigmoid()
activation_function_final = Sigmoid()

# Prepping for if you make us do He weight generation at any point!
weight_generator = GlorotUniform()

# Sliding window layer setup
for i in range(len(layer_window)-2):
	layers.append(Layer(layer_window[i], layer_window[i+1], activation_function_bulk, weight_generator))

layers.append(Layer(layer_window[-2], layer_window[-1], activation_function_final, weight_generator))

loss_function = CrossEntropy()

print("Making and Training Model")
model = MultilayerPerceptron(layers)
training_loss, validation_loss = model.train(x_train,
                                              y_train,
                                              x_val,
                                              y_val,
											  loss_function,
                                              epochs=50)

print("Plotting Results")
plt.plot(training_loss, color='b', label='Training')
plt.plot(validation_loss, color='r', label="Validation")
plt.title("Loss Curve", size=16)
plt.legend()
plt.show()

testing_predictions = model.forward(x_test)
testing_loss = loss_function.loss(testing_predictions, y_test)
print(f'Testing Loss: {testing_loss}')