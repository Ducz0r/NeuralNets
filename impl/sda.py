#-------------------------------------------------------------------------------
# Name:        impl.sda
# Purpose:     A Theano implementation of a stacked auto-encoder. Implemented after
#              the materials found on Deep Learning Tutorials (http://www.deeplearning.net/tutorial/SdA.html#sda).
#
# Author:      Luka Murn
#
# Created:     19. sep. 2013
# Copyright:   (c) Luka Murn 2013
#-------------------------------------------------------------------------------

# Imports
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy
import time

# Global variables (for debugging)
MODE = 'FAST_RUN' # 'FAST_COMPILE', 'FAST_RUN', 'DebugMode', 'ProfileMode'
#theano.config.compute_test_value = 'warn'

class StackedDenoisingAutoencoders(object):
    """
    Stacked denoising autoencoders implementation. Uses the sigmoid activation function on hidden layers.
    
    :param hidden_levels: number of nodes (w/o constant) in each hidden level
    :type hidden_levels: array
    
    :param in_dim: the dimensions of input
    :type in_dim: int
    
    :param learning_rate: learning rate for stochastic gradient descent
    :type learning_rate: float

    :param batch_size: the mini-batch size; if None, the net doesn't use mini-batches
    :type batch_size: int
    
    :param corruption_level: the % of corrupted attributes used when denoising the auto-encoders (value 0. or None represents no denoising)
    :type corruption_level: float
    
    :param n_epochs: maximal number of epochs to run the optimizer
    :type n_epochs: int
    
    :param seed: the random seed
    :type seed: int
    
    :param verbose: whether to output the optimization process
    :type verbose: boolean
    
    """
    
    def __init__(self, hidden_levels = [], in_dim = 5, learning_rate = 0.01,
                 batch_size = 20, corruption_level = None, n_epochs = 1000, seed = 1234,
                 verbose = True):
        self._hidden_levels = hidden_levels
        self._layer_sizes = numpy.concatenate([[in_dim], self._hidden_levels])
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._corruption_level = corruption_level if corruption_level is not None else 0.
        self._n_epochs = n_epochs
        self._seed = seed
        self._verbose = verbose
        self._trained = False
        
    def learn(self, train_data):
        """
        Learn the model (weights and biases) from the given data.
        
        :param train_data: the training set of examples, containing just the matrix X containing attribute values for individual examples
        :type train_data: matrix
        
        """
        #######################################
        # 0. INITIALIZE DATA SHARED VARIABLES #
        #######################################
        
        if self._verbose: print "Initializing shared data variables..."
        
        #########################
        # 2. BUILD ACTUAL MODEL #
        #########################
        
        if self._verbose: print "Building the model..."
        
        # Symbolic variables to represent data
        self._index = T.lscalar()  # index to a mini-batch, this will be the variable passed to Theano functions
        self._Xs = []  # the data matrix
        
        # Initialize random generators
        self._srng = RandomStreams(seed = self._seed) # we need this for corrupting the input (denoising)
        rng = numpy.random.RandomState(self._seed) # we need this to initialize weights (called only during initialization, therefore no need to execute it on GPU)
        
        # Initialize the weights and biases (array of 0's)
        self._weights = []
        self._biases = []
        self._biases_prime = []
        self._params = [] # Weights and biases represents the parameters of the learning model
        
        for i, n_out in enumerate(self._layer_sizes[1:]):
            i += 1
            n_in = self._layer_sizes[i - 1]
            
            # Weights for this level
            W_values = numpy.asarray(rng.uniform(low = -numpy.sqrt(6. / (n_in + n_out)), high = numpy.sqrt(6. / (n_in + n_out)), size = (n_in, n_out)), dtype = theano.config.floatX) * 4
            W = theano.shared(value = W_values, name = "W_" + str(i), borrow = True)
            self._weights.append(W)
            
            # Biases for this level
            b_values = numpy.zeros((n_out, ), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = "b_" + str(i), borrow = True)
            self._biases.append(b)
        
            # Prime biases for this level (for auto-encoders)
            b_values_prime = numpy.zeros((n_in, ), dtype = theano.config.floatX)
            b_prime = theano.shared(value = b_values_prime, name = "b_prime_" + str(i), borrow = True)
            self._biases_prime.append(b_prime)
            
            # Parameters for each layer
            self._params.append([W, b, b_prime])
        
        # Early-stopping parameters
        total_start_time = time.clock()
        previous_layer_output = train_data
        
        # LOOP OVER ALL LAYERS
        for layer in range(len(self._weights)):
            # CONSTRUCT SYMBOLIC FUNCTIONS
        
            # The following functions are the basis of computation
            X = T.matrix("X_" + str(layer))
            
            # Reconstruction function for auto-encoders
            # Firstly, we add some noise to the input
            input_ = self._srng.binomial(size = X.shape, n = 1, p = 1 - self._corruption_level) * X
            
            y = T.nnet.sigmoid(T.dot(input_, self._weights[layer]) + self._biases[layer])
            z = T.nnet.sigmoid(T.dot(y, self._weights[layer].T) + self._biases_prime[layer])
            
            code = y
            
            # Cost function for this level of auto-encoders
            L = -T.sum(X * T.log(z) + (1 - X) * T.log(1 - z), axis = 1)
            cost = T.mean(L)
            
            # Also, parameters for auto-encoders need to be set
            gparams = []
            for param in self._params[layer]:
                gparam = T.grad(cost, param)
                gparams.append(gparam)
            
                # DEFINE CONSTANT STUFF FOR THEANO FUNCTIONS
        
                # Inputs
                inputs_ = []
                if self._batch_size != None: inputs_.append(self._index)
        
            # Updates
            updates_ = []
            for param, gparam in zip(self._params[layer], gparams):
                updates_.append((param, param - self._learning_rate * gparam))
            
            ##################
            # 3. TRAIN MODEL #
            ##################
            
            if self._verbose: print "Training the model for layer " + str(layer + 1) + "..."
        
            start_time = time.clock()
        
            epoch = 1
            
            # Initialize shared variables (so we can copy all the data to the GPU
            # in one swipe to minimize the number of transfers)
            train_data_X = theano.shared(numpy.asarray(previous_layer_output, dtype = theano.config.floatX), borrow = True)
            
            # Compute the number of mini-batches for training
            n_train_batches = (train_data_X.get_value(borrow = True).shape[0] / self._batch_size) if self._batch_size != None else 1
            
            # Define the training function
            inputs_ = []
            if self._batch_size != None:
                inputs_.append(self._index)
            train_autoencoder = theano.function(inputs = inputs_, outputs = cost,
                                                updates = updates_,
                                                givens = {
                                                          X: train_data_X[self._index * self._batch_size : (self._index + 1) * self._batch_size] if self._batch_size != None else train_data_X[:]},
                                                mode = MODE)
            
            get_codes = theano.function(inputs = [], outputs = code,
                                        givens = {
                                                  X: train_data_X[:]},
                                        mode = MODE)
            
            # Main loop (epochs!)
            while epoch <= self._n_epochs:
                for minibatch_index in xrange(n_train_batches):
                    
                    # This is training phase
                    if self._batch_size != None:
                        minibatch_avg_cost = train_autoencoder(minibatch_index)
                    else:
                        minibatch_avg_cost = train_autoencoder()
                        
                    # Iteration number, this is the total number of trained minibatches
                    iter_ = (epoch - 1) * n_train_batches + minibatch_index + 1
        
                    if self._verbose: print('LAYER %i, Epoch %i, minibatch %i/%i (totally: %i), cost %f' %
                                  (layer + 1, epoch, minibatch_index + 1, n_train_batches, iter_, minibatch_avg_cost))
                    
                epoch += 1
            # End of main loop
            
            # We set the previous layer output as the full (all example) trained output of this layer
            previous_layer_output = get_codes()
            
            # Print the results of this layer's training
            end_time = time.clock()
            if self._verbose:
                print('Auto-encoder training for layer %i complete. Cost: %f. The code ran for %.2fm' % (layer + 1, minibatch_avg_cost, ((end_time - start_time) / 60.)))
            
        # Print the results
        total_end_time = time.clock()
        if self._verbose:
            print('Training auto-encoder optimization complete.')
            print('The code ran for %.2fm' % ((total_end_time - total_start_time) / 60.))

        # Alright, the weights are now trained!
        self._trained = True
        
    def get_parameters(self):
        """
        Get the parameters of the trained model.
        """
        if not self._trained:
            raise Exception("Cannot return parameters - the model is not yet trained!")
        return self._weights, self._biases, self._biases_prime
    
# Test code (on MNIST)
#import gzip, cPickle
#f = gzip.open("E:\\DIPLOMA\\Eclipse_workspace\\NeuralNets\\deep_learning_examples\\data\\mnist.pkl.gz", 'rb')
#train_set, _, _ = cPickle.load(f)
#f.close()
#sda = StackedDenoisingAutoencoders(hidden_levels = [500, 500], in_dim = 784, learning_rate = 0.001, batch_size = 10, corruption_level = 0.1, n_epochs = 15, seed = 89677, verbose = True)
#sda.learn(train_set[0])