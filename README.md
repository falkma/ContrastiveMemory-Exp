# ContrastiveMemory-Exp
Code for Falk*, Strupp*, et al. "Contrastive learning through non-equilibrium memory", to generate results in Fig. 3 (training an energy-based model to perform MNIST classification) .

Code is a lightly modified version of Ben Scellier's [Equilibrium Propagation codebase](https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop), which is written in [Theano](https://pypi.org/project/Theano/) (v1.0.5) with some auxillary Python (v2.7).

To produce data, have all files within the same directory and run: 
python train_convolution.py

Then plot_error_curves.py can be used to plot training progress.
