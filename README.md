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