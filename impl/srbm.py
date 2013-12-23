#-------------------------------------------------------------------------------
# Name:        impl.srbm
# Purpose:     A Theano implementation of stacked restricted Boltzmann machines (RBMs) that are a part of a DBN (deep belief net). Implemented after
#              the materials found on Deep Learning Tutorials (http://www.deeplearning.net/tutorial/rbm.html#rbm).
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

class StackedRestrictedBoltzmannMachines(object):
    """
    Stacked restricted Boltzmann machines (RBM) implementation. Use the sigmoid activation function on hidden layers.
    
    :param hidden_levels: number of nodes (w/o constant) in each hidden level
    :type hidden_levels: array
    
    :param in_dim: the dimensions of input
    :type in_dim: int
    
    :param learning_rate: learning rate for stochastic gradient descent
    :type learning_rate: float

    :param batch_size: the mini-batch size; if None, the net doesn't use mini-batches
    :type batch_size: int
    
    :param k: the k parameter for Gibbs sampling
    :type k: int
    
    :param persistent: if set to True, use persistent PCD
    :type persistent: boolean
    
    :param n_epochs: maximal number of epochs to run the optimizer
    :type n_epochs: int
    
    :param seed: the random seed
    :type seed: int
    
    :param verbose: whether to output the optimization process
    :type verbose: boolean
    
    """
    
    def __init__(self, hidden_levels = [], in_dim = 5, learning_rate = 0.01,
                 batch_size = 20, k = 1, persistent = False, n_epochs = 1000, seed = 1234,
                 verbose = True):
        self._hidden_levels = hidden_levels
        self._layer_sizes = numpy.concatenate([[in_dim], self._hidden_levels])
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._k = k
        self._persistent = persistent
        self._n_epochs = n_epochs
        self._seed = seed
        self._verbose = verbose
        self._trained = False
    
    ###################
    # RBM INNER CLASS #
    ###################
    class _RBM(object):
        """Restricted Boltzmann Machine (RBM).
        
        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.
    
        :param n_visible: number of visible units
        
        :param n_hidden: number of hidden units
        
        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP
        
        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network
        
        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias 
        
        """
        
        def __init__(self, input=None, n_visible=784, n_hidden=500,
                     W=None, hbias=None, vbias=None, numpy_rng=None,
                     theano_rng=None):
            self.n_visible = n_visible
            self.n_hidden = n_hidden


            if numpy_rng is None:
                # create a number generator
                numpy_rng = numpy.random.RandomState(1234)
            
            if theano_rng is None:
                theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
            if W is None:
                # W is initialized with `initial_W` which is uniformely sampled
                # from -4.*sqrt(6./(n_visible+n_hidden)) and 4.*sqrt(6./(n_hidden+n_visible))
                # the output of uniform if converted using asarray to dtype
                # theano.config.floatX so that the code is runable on GPU
                initial_W = numpy.asarray(numpy.random.uniform(
                         low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                         high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                         size=(n_visible, n_hidden)),
                         dtype=theano.config.floatX)
                # theano shared variables for weights and biases
                W = theano.shared(value=initial_W, name='W')
            
            if hbias is None :
                # create shared variable for hidden units bias
                hbias = theano.shared(value=numpy.zeros(n_hidden,
                                   dtype=theano.config.floatX), name='hbias')
            
            if vbias is None :
                # create shared variable for visible units bias
                vbias = theano.shared(value =numpy.zeros(n_visible,
                                    dtype = theano.config.floatX),name='vbias')
            
            
            # initialize input layer for standalone RBM or layer0 of DBN
            self.input = input if input else T.dmatrix('input')
            
            self.W = W
            self.hbias = hbias
            self.vbias = vbias
            self.theano_rng = theano_rng
            # **** WARNING: It is not a good idea to put things in this list
            # other than shared variables created in this function.
            self.params = [self.W, self.hbias, self.vbias]
        
        def propup(self, vis):
            """
            This function propagates the visible units activation upwards to
            the hidden units.
        
            Note that we return also the pre_sigmoid_activation of the layer. As
            it will turn out later, due to how Theano deals with optimization and
            stability this symbolic variable will be needed to write down a more
            stable graph (see details in the reconstruction cost function)
            """
            pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
            return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

        def sample_h_given_v(self, v0_sample):
            """This function infers state of hidden units given visible units."""
            # compute the activation of the hidden units given a sample of the visibles
            pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
            # get a sample of the hiddens given their activation
            # Note that theano_rng.binomial returns a symbolic sample of dtype
            # int64 by default. If we want to keep our computations in floatX
            # for the GPU we need to specify to return the dtype floatX
            h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean,
                                                 dtype=theano.config.floatX)
            return [pre_sigmoid_h1, h1_mean, h1_sample]
    
        def propdown(self, hid):
            """
            This function propagates the hidden units activation downwards to
            the visible units.
        
            Note that we return also the pre_sigmoid_activation of the layer. As
            it will turn out later, due to how Theano deals with optimization and
            stability this symbolic variable will be needed to write down a more
            stable graph (see details in the reconstruction cost function)
            """
            pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
            return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
    
        def sample_v_given_h(self, h0_sample):
            """This function infers state of visible units given hidden units."""
            # compute the activation of the visible given the hidden sample
            pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
            # get a sample of the visible given their activation
            # Note that theano_rng.binomial returns a symbolic sample of dtype
            # int64 by default. If we want to keep our computations in floatX
            # for the GPU we need to specify to return the dtype floatX
            v1_sample = self.theano_rng.binomial(size=v1_mean.shape,n=1, p=v1_mean,
                                                 dtype=theano.config.floatX)
            return [pre_sigmoid_v1, v1_mean, v1_sample]    
        
        def gibbs_hvh(self, h0_sample):
            """
            This function implements one step of Gibbs sampling,
            starting from the hidden state.
            """
            pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
            pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
            return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

        def gibbs_vhv(self, v0_sample):
            """
            This function implements one step of Gibbs sampling,
            starting from the visible state.
            """
            pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
            pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
            return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]
        
        def free_energy(self, v_sample):
            """Function to compute the free energy."""
            wx_b = T.dot(v_sample, self.W) + self.hbias
            vbias_term = T.dot(v_sample, self.vbias)
            hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
            return -hidden_term - vbias_term
        
        def get_cost_updates(self, lr=0.1, persistent=None, k=1):
            """
            This functions implements one step of CD-k or PCD-k.
            
            :param lr: learning rate used to train the RBM
            
            :param persistent: None for CD. For PCD, shared variable containing old state
            of Gibbs chain. This must be a shared variable of size (batch size, number of
            hidden units).
            
            :param k: number of Gibbs steps to do in CD-k/PCD-k
            
            Returns a proxy for the cost and the updates dictionary. The
            dictionary contains the update rules for weights and biases but
            also an update of the shared variable used to store the persistent
            chain, if one is used.
            
            """

            # compute positive phase
            pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        
            # decide how to initialize persistent chain:
            # for CD, we use the newly generate hidden sample
            # for PCD, we initialize from the old state of the chain
            if persistent is None:
                chain_start = ph_sample
            else:
                chain_start = persistent
                
            # perform actual negative phase
            # in order to implement CD-k/PCD-k we need to scan over the
            # function that implements one gibbs step k times.
            # Read Theano tutorial on scan for more information :
            # http://deeplearning.net/software/theano/library/scan.html
            # the scan will return the entire Gibbs chain
            [pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates = \
                theano.scan(self.gibbs_hvh,
                        # the None are place holders, saying that
                        # chain_start is the initial state corresponding to the
                        # 6th output
                        outputs_info=[None, None, None, None, None, chain_start],
                        n_steps=k)
                
            # determine gradients on RBM parameters
            # note that we only need the sample at the end of the chain
            chain_end = nv_samples[-1]
            
            cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
            # We must not compute the gradient through the gibbs sampling
            gparams = T.grad(cost, self.params, consider_constant=[chain_end])
            
            # constructs the update dictionary
            for gparam, param in zip(gparams, self.params):
                # make sure that the learning rate is of the right dtype
                updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
            if persistent:
                # Note that this works only if persistent is a shared variable
                updates[persistent] = nh_samples[-1]
                # pseudo-likelihood is a better proxy for PCD
                monitoring_cost = self.get_pseudo_likelihood_cost(updates)
            else:
                # reconstruction cross-entropy is a better proxy for CD
                monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])
            
            return monitoring_cost, updates
        
        def get_pseudo_likelihood_cost(self, updates):
            """Stochastic approximation to the pseudo-likelihood."""
            # index of bit i in expression p(x_i | x_{\i})
            bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        
            # binarize the input image by rounding to nearest integer
            xi = T.iround(self.input)
        
            # calculate free energy for the given bit configuration
            fe_xi = self.free_energy(xi)
        
            # flip bit x_i of matrix xi and preserve all other bits x_{\i}
            # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
            # the result to xi_flip, instead of working in place on xi.
            xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        
            # calculate free energy with bit flipped
            fe_xi_flip = self.free_energy(xi_flip)
        
            # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
            cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        
            # increment bit_i_idx % number as part of updates
            updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        
            return cost
        
        def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
            """
            Approximation to the reconstruction error
    
            Note that this function requires the pre-sigmoid activation as
            input.  To understand why this is so you need to understand a
            bit about how Theano works. Whenever you compile a Theano
            function, the computational graph that you pass as input gets
            optimized for speed and stability.  This is done by changing
            several parts of the subgraphs with others.  One such
            optimization expresses terms of the form log(sigmoid(x)) in
            terms of softplus.  We need this optimization for the
            cross-entropy since sigmoid of numbers larger than 30. (or
            even less then that) turn to 1. and numbers smaller than
            -30. turn to 0 which in terms will force theano to compute
            log(0) and therefore we will get either -inf or NaN as
            cost. If the value is expressed in terms of softplus we do not
            get this undesirable behaviour. This optimization usually
            works fine, but here we have a special case. The sigmoid is
            applied inside the scan op, while the log is
            outside. Therefore Theano will only see log(scan(..)) instead
            of log(sigmoid(..)) and will not apply the wanted
            optimization. We can not go and replace the sigmoid in scan
            with something else also, because this only needs to be done
            on the last step. Therefore the easiest and more efficient way
            is to get also the pre-sigmoid activation as an output of
            scan, and apply both the log and sigmoid outside scan such
            that Theano can catch and optimize the expression.
    
            """
            cross_entropy = T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                                         (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                                         axis=1))
            return cross_entropy
    ##########################
    # END OF RMB INNER CLASS #
    ##########################
    
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
        
        # Initialize a shared data variable
        train_data_X = theano.shared(numpy.asarray(train_data, dtype = theano.config.floatX), borrow = True)
        
        # Symbolic variables to represent data
        self._index = T.lscalar('index')  # index to a mini-batch, this will be the variable passed to Theano functions
        self._inputs = [T.matrix('X')]
        # Initialize random generators
        self._srng = RandomStreams(seed = self._seed) # we need this for corrupting the input (denoising)
        rng = numpy.random.RandomState(self._seed) # we need this to initialize weights (called only during initialization, therefore no need to execute it on GPU)
        
        # Initialize a list of RBMs (each represents one hidden layer)
        self._rbms = []
        
        for i, n_out in enumerate(self._layer_sizes[1:]):
            i += 1
            n_in = self._layer_sizes[i - 1]
            
            if i == 1:
                input_ = self._inputs[0]
            else:
                input_ = T.nnet.sigmoid(T.dot(self._inputs[-1], self._rbms[-1].W) + self._rbms[-1].hbias)
            self._inputs.append(input_)
            
            # Initialize a RBM for each layer
            rbm = self._RBM(numpy_rng = rng, theano_rng = self._srng, input = input_, n_visible = n_in, n_hidden = n_out)
            self._rbms.append(rbm)

            # Parameters for each layer
            #self._params.append(rbm.params)
        
        # CONSTRUCT THE PRE-TRAINING (SINGLE ITERATION STEP) GRADIENT DESCENT FUNCTIONS FOR EACH LAYER

        if self._batch_size is not None:
            # Begining of a batch, given `index`
            batch_begin = self._index * self._batch_size
            # ending of a batch given `index`
            batch_end = batch_begin + self._batch_size
        
        pretrain_fns = []
        for rbm in self._rbms:
            # Get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(lr = self._learning_rate, persistent = None, k = self._k)
        
            # Compile the Theano function; check if k is also a Theano
            # variable, if so added to the inputs of the function
            inputs_ = []
            if self._batch_size is not None:
                inputs_.append(self._index)
            if isinstance(self._k, theano.Variable):
                inputs_.append(self._k)
                
            fn = theano.function(inputs = inputs_,
                                 outputs = cost,
                                 updates = updates,
                                 givens = {
                                           self._inputs[0]: train_data_X[batch_begin : batch_end] if self._batch_size is not None else train_data_X[:]},
                                 mode = MODE)
            # Append `fn` to the list of functions
            pretrain_fns.append(fn)
        
        ##################
        # 3. TRAIN MODEL #
        ##################
        
        # Number of batches
        if self._batch_size is not None:
            n_batches = train_data.shape[0] / self._batch_size
        else:
            n_batches = 1
            
        # Early stopping parameters
        total_start_time = time.clock()
        
        # Pre-train each layer independently
        for i in xrange(len(self._rbms)):
            
            start_time = time.clock()
            if self._verbose: print "Training the model for layer " + str(i + 1) + "..."
            
            # go through pretraining epochs
            for epoch in xrange(self._n_epochs):
                
                # Go through the training set
                c = []
                for batch_index in xrange(n_batches):
                    if self._batch_size is not None:
                        c.append(pretrain_fns[i](index = batch_index))
                    else:
                        c.append(pretrain_fns[i]())
                    
                    # Iteration number, this is the total number of trained minibatches
                    iter_ = epoch * n_batches + batch_index + 1
                    
                    if self._verbose: print('LAYER %i, Epoch %i, minibatch %i/%i (totally: %i), cost %f' %
                                            (i + 1, epoch + 1, batch_index + 1, n_batches, iter_, numpy.mean(c)))

            # Print the results of this layer's training
            end_time = time.clock()
            if self._verbose:
                print('RBM training for layer %i complete. Cost: %f. The code ran for %.2fm' % (i + 1, numpy.mean(c), ((end_time - start_time) / 60.)))
        
        # Print the results
        total_end_time = time.clock()
        if self._verbose:
            print('Training RBMs optimization complete.')
            print('The code ran for %.2fm' % ((total_end_time - total_start_time) / 60.))

        # Alright, the weights are now trained!
        self._trained = True
        
    def get_parameters(self):
        """
        Get the parameters of the trained model.
        """
        if not self._trained:
            raise Exception("Cannot return parameters - the model is not yet trained!")
        return [rbm.W for rbm in self._rbms], [rbm.hbias for rbm in self._rbms], [rbm.vbias for rbm in self._rbms]
