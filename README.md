# Learning Flows using Neural ODEs

## Learning the dynamics of a double gyre system
This repository implements the learning of the dynamics of the double gyre system using Neural ODEs.
The model is trained using PyTorch[https://pytorch.org/docs/stable/index.html] and the torchdiffeq library[https://github.com/rtqichen/torchdiffeq].
Dynamics of the double gyre system are referenced from 'dgyre_eqns.pdf' (see the pdf in Neural_ODE/Reference). 
Detailed report of our experiments can be found in [ESE546_Project.pdf](ESE546_Project.pdf), also uploaded in the github repository

## Install
In run the code install:
```
torchdiffeq
numpy
torch
matplotlib
tqdm
```

## Training
To start, simply run [Neural_ODE/main.py](main.py).
To train the Neural ODE with prior-knowledge (KNODE):
1. Run Neural_ODE/main_invar_knode.py and set save_data as True to save the training data. (Invar is for time-invariant flow)
You can also play with the other parameters to see their effect on training.
For time varying flow our results were not that great but they can be run by making the desired changes in Neural_ODE/main.py

## Code Organisation
- The folder [Data](Data) contains the dataset extracted and used for training.
  It is extracted while running main.py and it can be used for additional visualization and testing.
- The folder [Images](Images) contains screenshots of the true and learned trajectories,
  in terms of time histories as well as phase portraits.
  These are taken during the training procedure.
  The number in the file titles represents the number of training epochs. (Currently a place holder on github as files are too large)
- The folder [Images_test](Images_test) contains screenshots of the true and learned trajectories during testing
- The folder [Models](Models) contains the trained model (Currently some folders contain models)

## Code Description
Description of files:
- [dynamics.py](dynamics.py): Contains the dynamic model for the double gyre system. (also see [dgyre_eqns.pdf](Reference/dgyre_eqns.pdf))
- [nn_models.py](nn_models.py): Contains the Neural ODE model for NODE
- [hybrid_nn_models.py](hybrid_nn_models.py) : Contains the Neural ODE model for KNODE
- [augmented_nn_models.py](augmented_nn_models.py) : Contains the Neural ODE model for ANODE
- [plotting_funcs.py](plotting_funcs.py): Contains functions required for plotting and visualization.
- [utils.py](utils.py): Utility functions for plotting, visualization etc.  
- [create_gif.py](create_gif.py): Creates a gif given images
- [test_knode.py]: script to test the generalizability of the trained knode model(written for time-invariant model). Similarly implemented for node and anode.

