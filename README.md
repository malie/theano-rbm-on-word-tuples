
A very simple Restricted Boltzmann Machine implementation.

Only CD-1
Shows Theano

Tries to learn the distribution of common word tuples from a text.



You'd need to install Theano and some python libs to get it running

Theano http://deeplearning.net/software/theano/

pip install scikit-learn
pip install gizeh



running the simplest version (that even hasn't minibatches) with

python rbm1.py


the minibatch version is in rbm-minibatch.py

the dropout version is in rbm-dropout.py but doesn't seem to work too well on this data.

