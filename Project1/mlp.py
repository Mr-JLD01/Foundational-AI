import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from ActivationsFunctions.ActivationFunction import ActivationFunction
from LossFunctions.LossFunction import LossFunction

from Weights.Weights import Weights
from Weights.GlorotUniform import GlorotUniform

def batch_generator(train_x: np.ndarray, train_y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    # needed the sanity check
    assert(train_x.shape[0] == train_y.shape[0])

    num_batches: int = train_x.shape[0] // batch_size + 1
    extra_batch: bool = train_x.shape[0] % batch_size != 0

    for i in range(num_batches):
        yield (train_x[i*batch_size : (i+1)*batch_size][:], train_y[i*batch_size : (i+1)*batch_size][:])

    # needed if batch_size doesn't cleanly divied
    if extra_batch:
        yield (train_x[num_batches*batch_size:][:], train_y[num_batches*batch_size:][:])


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, weight_generator: Weights):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in: int = fan_in
        self.fan_out: int = fan_out
        self.activation_function: ActivationFunction = activation_function

        # this will store the activations (forward prop)
        # the blank initialization is because this will get overwritten anyways
        self.activations:np.ndarray = np.zeros([])

        # this will store the delta term (dL_dPhi, backward prop)
        # the blank initialization is because this will get overwritten anyways
        self.delta:np.ndarray = np.zeros([])

        # Initialize weights and biases
        self.W: np.ndarray = weight_generator.generate(self.fan_in, self.fan_out) # weights
        # Had biases as ones before didn't seem to make a difference with zeros
        self.b: np.ndarray = np.zeros((1, self.fan_out), dtype=float)  # biases

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """

        # Needed to help find ways out of Dimensionality Hell
        assert(self.W.shape == (self.fan_in, self.fan_out))
        assert(self.b.shape == (1, self.fan_out))

        layer_before_activation: np.ndarray = h @ self.W + self.b

        self.activations = self.activation_function.forward(layer_before_activation)

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        dZ_dO = self.activation_function.derivative(self.activations)

        # print("input shape: {}".format(h.shape))
        # print("weight shape: {}".format(self.W.shape))
        # print("bias shape: {}".format(self.b.shape))
        # print("dZ_dO shape: {}".format(dZ_dO.shape))
        # print("delta shape: {}".format(delta.shape))

        # Functions taken from backprop diagram
        dL_dW = h.T @ (delta * dZ_dO)
        dL_db = delta * dZ_dO
        self.delta = (delta * dZ_dO) @ self.W.T #NOTE - May need fixing

        return (dL_dW, dL_db)


class MultilayerPerceptron:
    def __init__(self, layers: list[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers: list[Layer] = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """

        pred: np.ndarray = x

        for layer in self.layers:
            pred = layer.forward(pred)

        return pred

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all: list[np.ndarray] = []
        dl_db_all: list[np.ndarray] = []

        delta = loss_grad

        for i in range(len(self.layers)-1, -1, -1):
            if i == 0:
                dl_dw_n, dl_db_n = self.layers[i].backward(input_data, delta)
            else:
                dl_dw_n, dl_db_n = self.layers[i].backward(self.layers[i-1].activations, delta)

            delta = self.layers[i].delta

            # This is to more easily traverse in training method
            dl_dw_all.insert(0, dl_dw_n)
            dl_db_all.insert(0, dl_db_n)

        return (dl_dw_all, dl_db_all)

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """

        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            batches = batch_generator(train_x, train_y, batch_size)

            batch_loss = []

            for batch in batches:
                x, y_true = batch
                # Weird bug with batch generator that was giving an extra empty batch
                if len(x) == 0:
                    continue

                size_of_batch = len(x)
                y_pred = self.forward(x)

                delta = loss_func.derivative(y_true, y_pred)

                weight_gradients, bias_gradients = self.backward(delta, x)

                # another Dimensionality Hell check
                assert len(weight_gradients) == len(bias_gradients)

                for i in range(len(weight_gradients)):
                    self.layers[i].W = self.layers[i].W - ((learning_rate/size_of_batch) * np.sum(weight_gradients[i], axis=0))
                    self.layers[i].b = self.layers[i].b - ((learning_rate/size_of_batch) * np.sum(bias_gradients[i], axis=0))

                y_pred = self.forward(x)
                batch_loss.append(loss_func.loss(y_true, y_pred))

            t_loss = np.mean(np.array(batch_loss))
            training_losses.append(t_loss)

            val_y_pred = self.forward(val_x)
            v_loss = loss_func.loss(val_y, val_y_pred)
            validation_losses.append(v_loss)

            print("Epoch: {0} ::\tTraining Loss: {1}\t Validation loss: {2}".format(epoch, t_loss, v_loss))

        return np.array(training_losses), np.array(validation_losses)
