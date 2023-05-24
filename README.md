# Learning Flows using Neural ODEs
## Learning the double gyre system

This repository is built on top of the Learn_Double_Gyre repo at hsiehScalAR Lab GitHub repo implemented by KongYao Chee. It currently contains implementation of learning double gyre with Neural ODE.


This repository implements the learning of the dynamics of the double gyre system using Neural ODEs.
The training procedure is constructed using standard PyTorch tools (https://pytorch.org/docs/stable/index.html) and the torchdiffeq library (https://github.com/rtqichen/torchdiffeq).
Dynamics of the double gyre system are referenced from 'dgyre_eqns.pdf' (see attached pdf in #knodes channel, which is also in this repo).
For more details on the structure and parameters of the double gyre system, see dynamics.py.

---

To start, simply run [main.py](main.py).

---

Description of files:
- [dynamics.py](dynamics.py): Contains the dynamic model for the double gyre system. (also see [dgyre_eqns.pdf](Reference/dgyre_eqns.pdf))
- [nn_models.py](nn_models.py): Contains the Neural ODE models.
- [plotting_funcs.py](plotting_funcs.py): Contains functions required for plotting and visualization.
- utils.py: Utility functions.  
- The folder [Data](Data) contains the dataset extracted and used for training.
  It is extracted while running main.py, 
  and it can be used for additional visualization and testing.
- The folder [Images](Images) contains screenshots of the true and learned trajectories,
  in terms of time histories as well as phase portraits.
  These are taken during the training procedure.
  The number in the file titles represent the number of training epochs.

---

