import math
import random
from typing import Tuple
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
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

# fetch dataset
print("Fetching Data")
auto_mpg = fetch_ucirepo(id=9)

# data (as pandas dataframes)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine features and target into one DataFrame for easy filtering
data = pd.concat([X, y], axis=1)

# Drop rows where the target variable is NaN
print("Cleaning Data")
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

# Display the number of rows removed
rows_removed = len(data) - len(cleaned_data)
print(f"Rows removed: {rows_removed}")

print("Splitting Data")
# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,    # for reproducibility
    shuffle=True,       # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

print("Normalizing Data")
# Compute statistics for X (features)
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)    # Standard deviation of each feature

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y (targets)
y_mean = y_train.mean()  # Mean of target
y_std = y_train.std()    # Standard deviation of target

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# Here below is my own code minus the matplotlib code
print("Making Layers")
layer_window = []

# Need to account for vectors
num_inputs = X_train.shape[1] if len(X_train.shape) > 1 else 1
num_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1

layer_window.append(num_inputs)

# Layer sizes I get to choose
layer_window.append(20)
layer_window.append(10)
layer_window.append(15)
layer_window.append(50)
layer_window.append(10)
layer_window.append(15)

layer_window.append(num_outputs)

layers = []

activation_function_bulk = Sigmoid()
activation_function_final = Mish()

# Prepping for if you make us do He weight generation at any point!
weight_generator = GlorotUniform()

# Sliding window layer setup
for i in range(len(layer_window)-2):
	layers.append(Layer(layer_window[i], layer_window[i+1], activation_function_bulk, weight_generator))

layers.append(Layer(layer_window[-2], layer_window[-1], activation_function_final, weight_generator))

loss_function = SquaredError()

print("Making and Training Model")
model = MultilayerPerceptron(layers)
training_loss, validation_loss = model.train(X_train.to_numpy(),
                                              y_train.to_numpy(),
                                              X_val.to_numpy(),
                                              y_val.to_numpy(),
											  loss_function,
                                              epochs=200)

print("Plotting Results")
plt.plot(training_loss, color='b', label='Training')
plt.plot(validation_loss, color='r', label="Validation")
plt.title("Loss Curve", size=16)
plt.legend()
plt.show()

testing_predictions = model.forward(X_test.to_numpy())
testing_loss = loss_function.loss(testing_predictions, y_test.to_numpy())

# Wanted a neat table to screenshot for submission
print(f'Testing Loss: {testing_loss}')
print("Sample #|\t\tPred\t|\t\tTrue")
print("_"*70)
for _ in range(10):
	sample_number = random.randint(0, len(y_test))
	print(f"{sample_number}\t|\t{testing_predictions[sample_number]}\t|\t{y_test.to_numpy()[sample_number]}")
