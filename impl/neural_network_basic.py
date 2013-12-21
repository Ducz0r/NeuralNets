#-------------------------------------------------------------------------------
# Name:        Neural network algorithm implementation, made after Andrew Ng
#              and associated materials
#              (http://cs229.stanford.edu/materials.html). The neural network
#              contain regularization and logistic regression function in
#              the network nodes.
# Purpose:
#
# Author:      Luka Murn
#
# Created:     20. avg. 2013
# Copyright:   (c) Luka Murn 2013
#-------------------------------------------------------------------------------

# Imports
import numpy as np
import scipy.optimize.lbfgsb as scp
import random as rnd

class NeuralNetworkBasic(object):
    """
    A neural network learner/classifier implementation.

    :param hidden_levels: number of nodes (w/o constant) in each hidden level
    :type hidden_levels: array

    :param lambda_: the lambda parameter for regularization
    :type lambda_: float

    :param eps: the epsilon parameter for theta initialization
    :type eps: float
    
    :param seed: the random seed
    :type seed: float

    :param verbose: whether to output the optimization process of finding minimum of the cost function
    :type verbose: boolean

    :param opt_args: the additional parameters for optimization function
    :type opt_args: dictionary

    """

    def __init__(self, hidden_levels = [], lambda_ = 0., eps = 1e-4, seed = None, verbose = False, **opt_args):
        self._hl = hidden_levels
        self._lambda = lambda_
        self._eps = eps
        self._can_classify = False
        self._classes = []
        self._classes_map = {}
        self._verbose = verbose
        self._seed = seed
        if opt_args != None:
            self._opt_args = opt_args

    def learn(self, X, y):
        """
        Learn the model from the given data.

        :param X: the attribute data
        :type X: numpy.array

        :param y: the class variable data
        :type y: numpy.array

        """
        def rand(eps):
            """Return random number in interval [-eps, eps]."""
            return rnd.random() * 2 * eps - eps

        def g_func(z):
            """The sigmoid (logistic) function."""
            return 1. / (1. + np.exp(-z))

        def h_func(thetas, x):
            """The model function."""
            a = np.array([[1.] + list(x)]).T # Initialize a
            for l in range(1, len(thetas) + 1): # Forward propagation
                a = np.vstack((np.array([[1.]]), g_func(thetas[l - 1].T.dot(a))))
            return a[1:]

        def llog(val):
            "The limited logarithm."
            e = 1e-10
            return np.log(np.clip(val, e, 1. - e))
            
        def unroll(thetas):
            """Unrolls a list of thetas into vector."""
            sd = [m.shape for m in thetas] # Keep the shape data
            thetas = np.concatenate([theta.reshape(np.prod(theta.shape))for theta in thetas])
            return thetas, sd

        def roll(thetas, sd):
            """Rolls a vector of thetas back into list."""
            thetas = np.split(thetas, [sum(np.prod(s) for s in sd[:i]) + np.prod(sd[i]) for i in range(len(sd) - 1)])
            return [np.reshape(theta, sd[i]) for i, theta in enumerate(thetas)]

        def cost(thetas, X, y, sd, S, lambda_):
            """The cost function of the neural network."""
            thetas = roll(thetas, sd)
            m, _ = X.shape
            L = len(S)
            reg_factor = (lambda_ / float(2 * m)) * sum(sum(sum(thetas[l][1:, :]**2)) for l in range(L - 1))
            cost = (-1. / float(m)) * sum(sum(self._classes_map[y[s]] * llog(h_func(thetas, X[s])) + (1. - self._classes_map[y[s]]) * llog(1. - h_func(thetas, X[s]))) for s in range(m)) + reg_factor
            if self._verbose: print "Current value of cost func.: " + str(cost[0])
            return cost[0]
        
        def grad(thetas, X, y, sd, S, lambda_):
            """The gradient (derivate) function which includes the back
            propagation algorithm."""
            thetas = roll(thetas, sd)
            m, _ = X.shape
            L = len(S)
            d = [np.array([[0. for j in range(S[l + 1])] for i in range(S[l] + 1)]) for l in range(L - 1)] # Initialize the delta matrix
            
            for s in range(m):
                a = [np.array([[1.] + list(X[s])]).T] # Initialize a (only a, d & theta matrices have 1 more element in columns, biases)

                for l in range(1, L): # Forward propagation
                    a.append(np.vstack((np.array([[1.]]), g_func(thetas[l - 1].T.dot(a[l - 1])))))
                # TODO
                # Softmax: treat last a column differently
                #ez = np.exp(thetas[L - 2].T.dot(a[L - 2]))
                #sez = sum(ez)
                #a.append(np.vstack((np.array([[1]]), ez / sez)))
                
                deltas = [None for l in range(L - 1)] + [a[-1][1:] - self._classes_map[y[s]]]
                for l in range(L - 2, 0, -1): # Backward propagation
                    deltas[l] = (thetas[l].dot(deltas[l + 1]) * (a[l] * (1. - a[l])))[1:]

                for l in range(L - 1):
                    d[l] = d[l] + a[l].dot(deltas[l + 1].T)
            
            D = [(1. / float(m)) * d[l] + lambda_ * thetas[l] for l in range(L - 1)]
            D = [Di - lambda_ * np.vstack((thetas[l][0], np.zeros((Di.shape[0] - 1, Di.shape[1])))) for l, Di in enumerate(D)] # Where i = 0, don't use regularization
            D, _ = unroll(D)
            return D

        def gradApprox(thetas, X, y, sd, S, lambda_):
            """Approximate the gradient of the cost function
            (only used for debugging, not in final version)."""
            eps = 1e-14
            return (grad(thetas + eps, X, y, sd, S, lambda_) - grad(thetas - eps, X, y, sd, S, lambda_)) / float(2 * eps)
        
        # Set the random seed
        rnd.seed(self._seed)
        
        # Initialize the final layer of neural net (outputs)
        self._classes = list(set(y))
        for i, cl in enumerate(self._classes):
            self._classes_map[cl] = np.zeros((len(self._classes), 1))
            self._classes_map[cl][i] = 1.
            
        S = [len(X[0])] + self._hl + [len(self._classes)] # Complete information about levels
        L = len(S)
        thetas0 = [np.array([[rand(self._eps) for j in range(S[l + 1])] for i in range(S[l] + 1)]) for l in range(L - 1)] # Initialize the thetas matrix
        thetas0, sd = unroll(thetas0)
        #return grad(thetas0, X, y, sd, S, self._lambda), gradApprox(thetas0, X, y, sd, S, self._lambda) # For testing

        # The L-BFGS-B bounds parameter is redefined: input is (lower_bound, upper_bound) instead of array of bounds for each theta parameter
        if self._opt_args != None and "bounds" in self._opt_args:
            bounds = [self._opt_args["bounds"] for i in range(len(thetas0))]
            self._opt_args["bounds"] = bounds
            
        self._thetas, self._cost, _ = scp.fmin_l_bfgs_b(cost, thetas0, grad, args = (X, y, sd, S, self._lambda), **self._opt_args)
        self._thetas = roll(self._thetas, sd)
        self._cost = float(self._cost)
        self._can_classify = True

    def classify(self, x):
        """
        Classify the given data sample.

        :param x: the sample of data to be classified
        :type x: numpy.array

        """
        if not self._can_classify:
            raise Exception("Cannot classify if the model is not learned yet. Learn the model first.")
        thetas = self._thetas
        a = np.array([[1.] + list(x)]).T # Initialize a

        for l in range(1, len(thetas) + 1): # Forward propagation
            a = np.vstack((np.array([[1.]]), 1. / (1. + np.exp(-thetas[l - 1].T.dot(a)))))
        
        # TODO
        # SOFTMAX, find the class with max probability 
        class_probs = dict([(cl, a[i + 1][0]) for i, cl in enumerate(self._classes)])
        cl_sum = sum(class_probs.values())
        for cl in class_probs:
            class_probs[cl] = class_probs[cl] / float(cl_sum)
        cl = self._classes[np.argmax(a[1:])]
        return cl, class_probs

    def getModel(self):
        """
        Return the calculated model of neural network
        (thetas and value of cost function).

        """
        if not self._can_classify:
            raise Exception("Cannot return the model if it is not learned yet. Learn the model first.")
        return self._thetas, self._cost

    def setModel(self, thetas, cost):
        """
        Set the model via given thetas and cost.

        :param thetas: the thetas for the model
        :type thetas: array

        :param cost: the cost of the thetas function
        :type cost: float

        """
        self._thetas = thetas
        self._cost = cost
        self._can_classify = True