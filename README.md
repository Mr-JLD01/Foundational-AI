# Foundational-AI

The Code is ran within the [Project1](./Project1) directory.

To setup the environment, run the following commands inside the Project1 directory.

Setup a virtual environment. The requirements.txt is provided at the highest level. Once the environment is setup up, traverse to the Project1 folder. Here you can run either training file.
```bash
cd Project1

python3 train_mpg.py
# or
# NOTE - the mnist training currently overflows the loss and activation functions
python3 train_mnist.py
```

Within the Project1 folder, there are 3 custom libraries used in the code ActovationFunctions, LossFunctions, and Weights.

Weights is the library where all the weight generation functions are. The ones implemented are the following:
- Glorot Uniform

ActivationFunctions has the following functions:
- Relu
- Linear
- Mish
- Sigmoid
- Softplus
- Tanh
- Softmax* (Not finished)

LossFunctions has the following functions:
- SquaredError
- CrossEntropy