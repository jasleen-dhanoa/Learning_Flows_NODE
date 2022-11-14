## Learning Flows using Neural ODEs

This repository contains our code for learning the underlying flow using Neural ODEs. 

Code Organisation:-
The Neural_ODE folder contains the code from hsiehScalAR/learn_double_gyre for learning a time invariant double gyre using Neural ODE(Vanilla).

We also plan to learn the following:-
1. time invariant double gyre using Augmented Neural ODE
2. time invariant double gyre using KNODE
3. time invariant double gyre using RNN(Baseline)

So, we can have 4 folders with the above implementations. Since time varying double gyres will involve a chnage in dynamics equation they will have diffrent files in the above folders.
We can have another folder for SINDy if we end up doing sparse identification of non-linear dynamical systems to obtain an analytical model.

Goals:-
1. Learn time invariant double gyre for:-

- [ ] Neural ODE
- [ ] Augmented Neural ODE
iii) KNODE
iv) RNN
2. Compare the performances
3. Learn time varying double gyre for:-
i) Neural ODE
ii) Augmented Neural ODE
iii) KNODE
iv) RNN
4. Compare the performances
5. Learn the analytical model
6. Write Report and discussion


