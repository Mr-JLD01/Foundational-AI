# This file serves as the testing ground/debug playground for all functions made

import numpy as np

from ActivationsFunctions.ActivationFunction import ActivationFunction
from ActivationsFunctions.Linear import Linear
from ActivationsFunctions.Mish import Mish
from ActivationsFunctions.Relu import Relu
from ActivationsFunctions.Sigmoid import Sigmoid
from ActivationsFunctions.Softmax import Softmax
from ActivationsFunctions.Softplus import Softplus
from ActivationsFunctions.Tanh import Tanh

from LossFunctions.SquaredError import SquaredError
from LossFunctions.CrossEntropy import CrossEntropy

from Weights.GlorotUniform import GlorotUniform

from mlp import batch_generator

test_array = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
test_loss_array_1 = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
test_loss_array_2 = np.array([-7,-4,4,99,-1,4,1,2,11,4,10])

def test_print(test_arr: np.ndarray, func: ActivationFunction, name: str) -> None:
	print(name)
	print("Forward")
	print(func.forward(test_arr))
	print("Derivative")
	print(func.derivative(test_arr))
	print("="*50)
	print()


# test_print(test_array, Linear(), "Linear")
# test_print(test_array, Relu(), "Relu")
# test_print(test_array, Sigmoid(), "Sigmoid")
# test_print(test_array, Tanh(), "Tanh")
test_print(test_array, Softmax(), "Softmax")
# test_print(test_array, Softplus(), "Softplus")
# test_print(test_array, Mish(), "Mish")

larger_true = np.array([test_array, test_array])
print(larger_true.size)
print(larger_true.shape)
print(len(larger_true))
larger_pred = np.array([test_loss_array_1, test_loss_array_2])
print("Loss")
se = SquaredError()
print(se.loss(test_array, test_loss_array_2))
print(se.derivative(test_array, test_loss_array_2))
print(se.loss(larger_true, larger_pred))
print(se.derivative(larger_true, larger_pred))

# print("Loss 2")
# ce = CrossEntropy()
# print(ce.loss(test_array, test_loss_array_2))
# print(ce.loss(larger_true, larger_pred))

print("Weights")
gu = GlorotUniform()
print(gu.generate(2,3))

x = test_array.T.reshape(11,1) * test_array
print(x)

batches = batch_generator(x, test_loss_array_1.T, 10)

for batch in batches:
	print(batch)
