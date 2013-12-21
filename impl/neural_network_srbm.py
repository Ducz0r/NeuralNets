#-------------------------------------------------------------------------------
# Name:        impl.neural_network_srbm
# Purpose:     A Theano implementation of a neural network with stacking restricted Boltzmann machines.
#
# Author:      Luka Murn
#
# Created:     2. sep. 2013
# Copyright:   (c) Luka Murn 2013
#-------------------------------------------------------------------------------

# Imports
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy
import time
import cPickle
from impl.srbm import StackedRestrictedBoltzmannMachines

# Global variables (for debugging)
MODE = 'FAST_RUN' # 'FAST_COMPILE', 'FAST_RUN', 'DebugMode', 'ProfileMode'
#theano.config.compute_test_value = 'warn'

class NeuralNetworkWithSrbm(object):
    """
    A neural network learner/classifier implementation. Uses the specified activation function on hidden layers, and softmax on the last level.
    Also uses the stacked denoising auto-encoders for weight pre-training.

    :param hidden_levels: number of nodes (w/o constant) in each hidden level; if empty array, the neural net becomes a simple softmax log. regression
    :type hidden_levels: array
    
    :param in_out: the dimensions of input and output (nr. of attributes for specific sample and nr. of different classes/labels)
    :type in_out: tuple (int, int)
    
    :param learning_rate: learning rate for stochastic gradient descent
    :type learning_rate: float

    :param L1_reg: the lambda parameter for L1 regularization
    :type L1_reg: float

    :param L2_reg: the lambda parameter for L2 regularization
    :type L2_reg: float
    
    :param dropout_thresholds: thresholds for dropout (inputs_thresh, thresh) where the values represent possibility the unit will output activity - if (1., 1.), there is no dropout
    :type dropout_thresholds: tuple (float, float)
    
    :param batch_size: the mini-batch size; if None, the net doesn't use mini-batches
    :type batch_size: int
    
    :param n_epochs: maximal number of epochs to run the optimizer
    :type n_epochs: int

    :param seed: the random seed
    :type seed: int
    
    :param es_patience: early stopping - look as this many examples regardless
    :type es_patience: int
    
    :param es_patience_inc: early stopping - wait this much longer when a new best is found
    :type es_patience_inc: int
    
    :param es_improvement_thresh: early stopping - a relative improvement of this much is considered significant
    :type es_improvement_thresh: float
    
    :param srbm_learning_rate: learning rate for training stacked restricted Boltzmann machines
    :type srbm_learning_rate: float
    
    :param srbm_batch_size: the mini-batch size for weight pre-training; if None, no minibatches are used
    :tyoe srbm_batch_size: int
    
    :param srbm_k: the number of steps for Gibbs sampling in restricted Boltzmann machines weight pre-training
    :type srbm_k: int
    
    :param srbm_persistent: if True, use persistent Gibbs sampling (PCD)
    :type srbm_persistent: boolean
    
    :param srbm_n_epochs: maximal number of epochs to run the weight initializer
    :type srbm_n_epochs: int
    
    :param verbose: whether to output the optimization process
    :type verbose: boolean

    """
    
    def __init__(self, hidden_levels = [], in_out = (5, 5), learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001,
                 dropout_thresholds = (1., 1.), batch_size = 20, n_epochs = 1000, seed = 1234,
                 es_patience = 10000, es_patience_inc = 2, es_improvement_thresh = 0.995, 
                 srbm_learning_rate = 0.01, srbm_batch_size = 20, srbm_k = 1, srbm_persistent = False, srbm_n_epochs = 500, verbose = True):
        self._hidden_levels = hidden_levels
        self._layer_sizes = numpy.concatenate([[in_out[0]], self._hidden_levels, [in_out[1]]])
        self._learning_rate = learning_rate
        self._L1_reg = L1_reg
        self._L2_reg = L2_reg
        self._dropout_thresh_input = dropout_thresholds[0]
        self._dropout_thresh = dropout_thresholds[1]
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._activation = T.nnet.sigmoid
        self._seed = seed
        self._es_patience = es_patience
        self._es_patience_inc = es_patience_inc
        self._es_improvement_thresh = es_improvement_thresh
        self._srbm_learning_rate = srbm_learning_rate
        self._srbm_batch_size = srbm_batch_size
        self._srbm_k = srbm_k
        self._srbm_persistent = srbm_persistent
        self._srbm_n_epochs = srbm_n_epochs
        self._verbose = verbose
        self._trained = False
        
    def learn(self, train_set, valid_set, test_set = None):
        """
        Learn the model from the given data.
        
        :param train_set: the training set of examples, containing both the examples as X matrix and the classes as y vector
        :type train_set: tuple (matrix, vector)
        
        :param valid_set: the set of examples used for validation (stochastic gradient descent!), same format as train_set
        :type valid_set: tuple (matrix, vector)
        
        :param test_set: the set of examples used for testing the model score, can be set to None, same format as train_set
        :type test_set: tuple (matrix, vector)
        
        """
        def shared_dataset(data_xy, borrow = True):
            """Function that loads the dataset into shared variables, so they can 
            be stored on the GPU at all times and in one transfer."""
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
            shared_y = theano.shared(numpy.asarray(data_y, dtype = theano.config.floatX), borrow = borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, "int32")
        
        ###################################
        # 0. INITIALIZE CERTAIN VARIABLES #
        ###################################
        
        if self._verbose: print "Initializing variables..."
        
        # Test set can be None, in that case we don't use test data
        use_test_set = test_set != None
        
        # Retrieve the classes in the training set, construct mapping from 'class' -> 'int'
        # (GPU can only work with numbers, actually floats)
        self._classes = list(set(train_set[1]))
        self._class_map = dict((cl, i) for i, cl in enumerate(self._classes))
        self._class_map_reverse = dict((i, cl) for i, cl in enumerate(self._classes))
        
        # Re-format the class (y) arrays (we must convert string classes/labels to integers)
        train_set = (train_set[0], [self._class_map[cl] for cl in train_set[1]])
        valid_set = (valid_set[0], [self._class_map[cl] for cl in valid_set[1]])
        if use_test_set:
            test_set = (test_set[0], [self._class_map[cl] for cl in test_set[1]])
        
        #######################################
        # 1. INITIALIZE DATA SHARED VARIABLES #
        #######################################
        
        if self._verbose: print "Initializing shared data variables..."
        
        # Initialize shared variables (so we can copy all the data to the GPU
        # in one swipe to minimize the number of transfers)
        train_set_X, train_set_y = shared_dataset(train_set)
        valid_set_X, valid_set_y = shared_dataset(valid_set)
        if use_test_set:
            test_set_X, test_set_y = shared_dataset(test_set)
        
        # Compute the number of mini-batches for training, validation and 
        # possibly testing
        n_train_batches = (train_set_X.get_value(borrow = True).shape[0] / self._batch_size) if self._batch_size != None else 1
        n_valid_batches = (valid_set_X.get_value(borrow = True).shape[0] / self._batch_size) if self._batch_size != None else 1
        if use_test_set:
            n_test_batches = (test_set_X.get_value(borrow = True).shape[0] / self._batch_size) if self._batch_size != None else 1
        
        #########################
        # 2. BUILD ACTUAL MODEL #
        #########################
        
        if self._verbose: print "Building the model..."
        
        # Symbolic variables to represent data
        self._index = T.lscalar()  # index to a mini-batch, this will be the variable passed to Theano functions
        self._X = T.matrix("X")  # the data matrix
        self._y = T.ivector("y")  # the labels are presented as 1D vector of int values
        self._d = T.scalar("d", dtype = theano.config.floatX) # the dropout threshold (used in forward propagation for training when using dropout)
        self._di = T.scalar("di", dtype = theano.config.floatX) # the dropout threshold for input values (used in forward propagation for training when using dropout)
        self._dm = T.scalar("dm", dtype = theano.config.floatX) # the multiplier of weights (used in forward propagation for testing, classifying when using dropout)
        self._dmi = T.scalar("dmi", dtype = theano.config.floatX) # the multiplier of weights for input values (used in forward propagation for testing, classifying when using dropout)
        
        # Initialize random generators
        self._srng = RandomStreams(seed = self._seed) # we need this for dropout (every time forward propagation is done in training we randomly drop out some units, executed on GPU)
        rng = numpy.random.RandomState(self._seed) # we need this to initialize weights (called only during initialization, therefore no need to execute it on GPU)
        
        # Initialize the weights (special handling is taken if using sigmoid function) and biases (array of 0's)
        self._weights = []
        self._biases = []
        self._params = [] # Weights and biases represents the parameters of the classifying (neural net) model
        
        # Aaalright, let's pre-train the weights :)
        # Construct the denoising auto-encoders
        srbm = StackedRestrictedBoltzmannMachines(hidden_levels = self._hidden_levels, in_dim = self._layer_sizes[0], learning_rate = self._srbm_learning_rate, 
                                                  batch_size = self._srbm_batch_size, k = self._srbm_k, persistent = self._srbm_persistent, 
                                                  n_epochs = self._srbm_n_epochs, seed = self._seed, verbose = self._verbose)
        
        # Learn the denoising auto-encoders
        if self._verbose: 
            print "====================================================================="
            print "= Pre-training weights & biases using restricted Boltzmann machines ="
            print "====================================================================="
        
        srbm.learn(train_set[0])
        ws, bs, _ = srbm.get_parameters()
        
        if self._verbose: 
            print "=============================="
            print "= End of weight pre-training ="
            print "=============================="
        
        # Set the weights and biases from the pre-trained model
        for i in range(len(ws)):
            W = ws[i]
            b = bs[i]
            
            self._weights.append(W)
            self._params.append(W)
            
            self._biases.append(b)
            self._params.append(b)
        
        # Set the weights and biases for the last, softmax layer!
        n_in = self._layer_sizes[-2]
        n_out = self._layer_sizes[-1]
        W_values = numpy.asarray(rng.uniform(low = -numpy.sqrt(6. / (n_in + n_out)), high = numpy.sqrt(6. / (n_in + n_out)), size = (n_in, n_out)), dtype = theano.config.floatX)
        if self._activation == theano.tensor.nnet.sigmoid:
            W_values *= 4
        W = theano.shared(value = W_values, name = "W_" + str(i), borrow = True)
        self._weights.append(W)
        self._params.append(W)
        
        b_values = numpy.zeros((n_out, ), dtype = theano.config.floatX)
        b = theano.shared(value = b_values, name = "b_" + str(i), borrow = True)
        self._biases.append(b)
        self._params.append(b)
        
        # Now that we have weights, we can construct weight decay sums
        self._L1 = numpy.sum(abs(W).sum() for W in self._weights)
        self._L2_sqr = numpy.sum((W ** 2).sum() for W in self._weights)
        
        # CONSTRUCT SYMBOLIC FUNCTIONS
        
        # Construct the output Theano function (for all but the last 
        # layer, this will be a recursive call of activation functions) - 
        # this is actually the FORWARD PROPAGATION
        
        # First level, we just dropout (if needed) some input values
        outputs = [self._X]
        outputs[0] *= self._dmi
        outputs[0] *= T.le(self._srng.uniform(size = self._X.shape, dtype = theano.config.floatX), self._di)
        
        for i in range(len(self._weights) - 1):
            # When classifying, we need to multiply the weights by e.g. 0.5 if using dropout; 
            # otherwise, we just multiply them by 1.
            lin_output = T.dot(outputs[i], self._weights[i]) + self._biases[i]
            output = (lin_output if self._activation is None else self._activation(lin_output)) * self._dm
            
            # When training, we randomly drop out some output activities if using dropout; otherwise, threshold is 1.
            # and every random number in [0., 1.) is less than that, so we don't drop out any outputs
            dropout_filter = T.le(self._srng.uniform(size = output.shape, dtype = theano.config.floatX), self._d)
            
            outputs.append(output * dropout_filter)
        
        # The following 4 functions are the basis of computation
        
        # Last layer is always softmax
        self._p_y_given_x = T.nnet.softmax(T.dot(outputs[-1], self._weights[-1]) + self._biases[-1])
        self._y_pred = T.argmax(self._p_y_given_x, axis = 1)
        
        # Initialize the cost function (negative log likelihood) we wish to minimize
        self._cost = -T.mean(T.log(self._p_y_given_x)[T.arange(self._y.shape[0]), self._y]) + self._L1_reg * self._L1 + self._L2_reg * self._L2_sqr
        #self._cost = T.mean(T.nnet.categorical_crossentropy(self._p_y_given_x, self._y))
        
        # Initialize the errors function
        self._errors = T.mean(T.neq(self._y_pred, self._y))
        
        # Compute the gradients of cost function with respect to weights and biases
        self._gparams = []
        for param in self._params:
            gparam = T.grad(self._cost, param)
            self._gparams.append(gparam)
        
        # DEFINE ACTUAL, CALLABLE FUNCTIONS
        
        # (we use ignore inputs parameter because if we have only one hidden layer, we can't really do dropout as the 
        # last layer is always softmax layer, so the dropout inputs end up not being used)
        
        # Inputs are the same for all functions
        inputs_ = [self._d, self._di, self._dm, self._dmi]
        if self._batch_size != None: inputs_.append(self._index)
        
        # Compile 2 Theano functions that compute the mistakes that are made
        # by the model on a mini-batch
        if use_test_set:
            test_model = theano.function(inputs = inputs_,
                outputs = self._errors,
                givens = {
                    self._X: test_set_X[self._index * self._batch_size : (self._index + 1) * self._batch_size] if self._batch_size != None else test_set_X[:],
                    self._y: test_set_y[self._index * self._batch_size : (self._index + 1) * self._batch_size] if self._batch_size != None else test_set_y},
                on_unused_input = 'ignore',
                mode = MODE)
        
        validate_model = theano.function(inputs = inputs_,
            outputs = self._errors,
            givens = {
                self._X: valid_set_X[self._index * self._batch_size : (self._index + 1) * self._batch_size] if self._batch_size != None else valid_set_X[:],
                self._y: valid_set_y[self._index * self._batch_size : (self._index + 1) * self._batch_size] if self._batch_size != None else valid_set_y},
            on_unused_input = 'ignore',
            mode = MODE)
        
        # Specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates_ = []
        for param, gparam in zip(self._params, self._gparams):
            updates_.append((param, param - self._learning_rate * gparam))
        
        # Compile a Theano function "train_model" that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in "updates"
        train_model = theano.function(inputs = inputs_, outputs = self._cost,
                updates = updates_,
                givens = {
                    self._X: train_set_X[self._index * self._batch_size : (self._index + 1) * self._batch_size] if self._batch_size != None else train_set_X[:],
                    self._y: train_set_y[self._index * self._batch_size : (self._index + 1) * self._batch_size] if self._batch_size != None else train_set_y},
                on_unused_input = 'ignore',
                mode = MODE)
        
        ##################
        # 3. TRAIN MODEL #
        ##################
        
        if self._verbose: print "Training the model..."
        
        # Early-stopping parameters
        validation_frequency = min(n_train_batches, self._es_patience / 2)
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
    
        epoch = 0
        done_looping = False
    
        # Main loop (epochs!)
        while (epoch < self._n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                
                # This is training phase; we dropout some units, but the weight multiplier is 1
                if self._batch_size != None:
                    minibatch_avg_cost = train_model(self._dropout_thresh, self._dropout_thresh_input, 1., 1., minibatch_index)
                else:
                    minibatch_avg_cost = train_model(self._dropout_thresh, self._dropout_thresh_input, 1., 1.)
                    
                # Iteration number, this is the total number of trained minibatches
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
                    # Compute zero-one loss on validation set
                    # (here, we don't dropout any units, but we multiply the weights by a threshold!)
                    validation_losses = [validate_model(1., 1., self._dropout_thresh, self._dropout_thresh_input, i) if self._batch_size != None else 
                                         validate_model(1., 1., self._dropout_thresh, self._dropout_thresh_input) for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
    
                    if self._verbose: 
                        print('Epoch %i, minibatch %i/%i (totally: %i), validation error %f %%' %
                              (epoch, minibatch_index + 1, n_train_batches, iter, this_validation_loss * 100.))
    
                    # If we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # Improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * self._es_improvement_thresh:
                            self._es_patience = max(self._es_patience, iter * self._es_patience_inc)
    
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
                        # Test it on the test set, if available
                        # (here, we don't dropout any units, but we multiply the weights by a threshold!)
                        if use_test_set:
                            test_losses = [test_model(1., 1., self._dropout_thresh, self._dropout_thresh_input, i) if self._batch_size != None else 
                                           test_model(1., 1., self._dropout_thresh, self._dropout_thresh_input) for i in xrange(n_test_batches)]
                            test_score = numpy.mean(test_losses)
                            
                            if self._verbose:
                                print(('     Epoch %i, minibatch %i/%i (totally: %i), test error of best model %f %%') %
                                      (epoch, minibatch_index + 1, n_train_batches, iter, test_score * 100.))
    
                if self._es_patience <= iter:
                        done_looping = True
                        break
        # End of main loop
        
        # Print the results
        end_time = time.clock()
        if self._verbose:
            print(('Training optimization complete. Best validation score of %f %% obtained at iteration %i, with test performance %f %%') %
                  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
            print('The code ran for %.2fm' % ((end_time - start_time) / 60.))

        # Neural net is now ready for classification!
        self._trained = True

    def classify(self, x, return_all_probs = False):
        """
        Classify the given data sample.
        
        :param x: the sample of data to be classified
        :type x: numpy.array
        
        :param return_all_probs: if True, return the probabilities of all classes
        :type return_all_probs: boolean
        
        """
        if not self._trained:
            raise Exception("Cannot classify - the model is not yet trained!")
        
        shared_x = theano.shared(numpy.asarray([x], dtype = theano.config.floatX), borrow = True)
        forward_prop = theano.function(
                inputs = [self._d, self._di, self._dm, self._dmi], 
                outputs = [self._y_pred, self._p_y_given_x],
                givens = { self._X: shared_x },
                on_unused_input = 'ignore',
                mode = MODE)
        cl_index, cl_vars = forward_prop(1., 1., self._dropout_thresh, self._dropout_thresh_input)
        if return_all_probs:
            return dict((self._class_map_reverse[i], val) for i, val in enumerate(cl_vars[0]))
        else:
            return self._class_map_reverse[cl_index[0]], cl_vars[0][cl_index[0]]

    def classify_all(self, X, return_all_probs = False):
        """
        Classify multilpe data samples (in order to minimize the data transfers to the GPU).
        
        :param X: the matrix of data samples to be classified
        :type x: numpy.array
        
        :param return_all_probs: if True, return the probabilities of all classes
        :type return_all_probs: boolean
        
        """
        if not self._trained:
            raise Exception("Cannot classify - the model is not yet trained!")
        
        shared_X = theano.shared(numpy.asarray(X, dtype = theano.config.floatX), borrow = True)
        forward_prop = theano.function(
                inputs = [self._d, self._di, self._dm, self._dmi], 
                outputs = [self._y_pred, self._p_y_given_x],
                givens = { self._X: shared_X },
                on_unused_input = 'ignore',
                mode = MODE)
        cl_indexes, cl_vars = forward_prop(1., 1., self._dropout_thresh, self._dropout_thresh_input)
        if return_all_probs:
            return [dict((self._class_map_reverse[j], val) for j, val in enumerate(cl_vars[i])) for i in range(len(cl_indexes))]
        else:
            return [(self._class_map_reverse[cl_indexes[i]], cl_vars[i][cl_indexes[i]]) for i in range(len(cl_indexes))]
        
    def save_model_to_file(self, file_path):
        """
        Save the model parameters to the specified file.
        
        :param file_path: The absolute path to the file
        :type file_path: string
        
        """
        if self._trained == False:
            raise Exception("Cannot save the model if it hasn't been trained yet!")
        save_file = open(file_path, 'wb')
        cPickle.dump(len(self._weights), save_file)
        cPickle.dump(self._class_map, save_file)
        cPickle.dump(self._class_map_reverse, save_file)
        for W in self._weights:
            cPickle.dump(W.get_value(borrow = True), save_file)
        for b in self._biases:
            cPickle.dump(b.get_value(borrow = True), save_file)
        save_file.close()

    def load_model_from_file(self, file_path):
        """
        Load the model parameters from the specified file. The model parameters must match 
        those set in the initialization of the neural net (hidden layer sizes).
        
        :param file_path: The absolute path to the file
        :type file_path: string
        
        """
        # Firstly, load the parameters (weights, biases)
        load_file = open(file_path)
        len_W = cPickle.load(load_file)
        if len_W != len(self._hidden_levels) + 1:
            raise Exception("Incorrect model parameters in file (hidden levels)!")
        
        self._class_map = cPickle.load(load_file)
        self._class_map_reverse = cPickle.load(load_file)
        
        self._weights = []
        self._biases = []
        self._params = []
        for i in range(len_W):
            W_vals = cPickle.load(load_file)
            if W_vals.shape != (self._layer_sizes[i], self._layer_sizes[i + 1]):
                raise Exception("Incorrect model parameters in file (weights)!")
            W = theano.shared(value = W_vals, name = "W_" + str(i), borrow = True)
            self._weights.append(W)
            self._params.append(W)
        for i in range(len_W):
            b_vals = cPickle.load(load_file)
            if len(b_vals) != self._layer_sizes[i + 1]:
                raise Exception("Incorrect model parameters in file (biases)!")
            b = theano.shared(value = b_vals, name = "b_" + str(i), borrow = True)
            self._biases.append(b)
            self._params.append(b)
        load_file.close()
        
        # Re-define the forward propagation function, so you can use it for classification
        # (that means we need to make sure all the symbolic variables are initialized as well)
        self._X = T.matrix("X")
        self._y = T.ivector("y")
        self._d = T.fscalar("d")
        self._di = T.fscalar("di")
        self._dm = T.fscalar("dm")
        self._dmi = T.fscalar("dmi")
        self._srng = RandomStreams(seed=self._seed)
        outputs = [self._X]
        outputs[0] *= self._dmi
        outputs[0] *= T.le(self._srng.uniform(size = self._X.shape, dtype = theano.config.floatX), self._di)
        for i in range(len(self._weights) - 1):
            # When classifying, we need to multiply the weights by e.g. 0.5 if using dropout; 
            # otherwise, we just multiply them by 1.
            lin_output = T.dot(outputs[i], self._weights[i]) + self._biases[i]
            output = (lin_output if self._activation is None else self._activation(lin_output)) * self._dm
            
            # When training, we randomly drop out some output activities if using dropout; otherwise, threshold is 1.
            # and every random number in [0., 1.) is less than that, so we don't drop out any outputs
            dropout_filter = T.le(self._srng.uniform(size = output.shape, dtype = theano.config.floatX), self._d)
            
            outputs.append(output * dropout_filter)
        # Last layer is always softmax
        self._p_y_given_x = T.nnet.softmax(T.dot(outputs[-1], self._weights[-1]) + self._biases[-1])
        self._y_pred = T.argmax(self._p_y_given_x, axis = 1)

        self._trained = True

# Test code (on MNIST)
#import gzip, cPickle
#f = gzip.open("E:\\DIPLOMA\\Eclipse_workspace\\NeuralNets\\deep_learning_examples\\data\\mnist.pkl.gz", 'rb')
#train_set, valid_set, test_set = cPickle.load(f)
#f.close()
#nn = NeuralNetworkWithSrbm(hidden_levels = [200, 200], in_out = (784, 10), dropout_thresholds = (0.8, 0.5), batch_size = 200, seed = 42, learning_rate = 0.2, n_epochs = 1500,
#                          srbm_learning_rate = 0.05, srbm_batch_size = 200, srbm_k = 1, srbm_persistent = False, srbm_n_epochs = 15)
#nn.learn(train_set, valid_set, test_set)
#nn.save_model_to_file("E:\\logreg.pickle")
#nn.load_model_from_file("E:\\logreg.pickle")