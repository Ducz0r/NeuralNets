#-------------------------------------------------------------------------------
# Name:        test
# Purpose:     A module containing some sample calls on how to use the library.
#
# Author:      Luka Murn
#
# Created:     21. dec. 2013
# Copyright:   (c) Luka Murn 2013
#-------------------------------------------------------------------------------

# Global imports
import numpy as np

# =======================
# CODE:
# =======================

# # Test the basic neural network
# from impl.neural_network_basic import NeuralNetworkBasic
# X = np.array([[1., 2., 3.], [5., 6., 7.]])
# y = np.array([0., 1.])
# nn = NeuralNetworkBasic([3, 2], 2.)
# nn.learn(X, y)
# a, b = nn.getModel()

# Test the Theano neural network on MNIST
from impl.neural_network import NeuralNetwork, rectified_linear, tanh_modified
import gzip, cPickle
f = gzip.open("D:\\DIPLOMA\\Eclipse_workspace\\NeuralNets\\deep_learning_examples\\data\\mnist.pkl.gz", 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
nn = NeuralNetwork(hidden_levels = [200, 200], in_out = (784, 10), activation = rectified_linear, dropout_thresholds = (0.8, 0.5), batch_size = 200, seed = 42, learning_rate = 0.2, n_epochs = 1500)
nn.learn(train_set, valid_set, test_set)
nn.save_model_to_file("E:\\model.pickle")
nn.load_model_from_file("E:\\model.pickle")

# # Test the Theano neural network with SDA on MNIST
# from impl.neural_network_sda import NeuralNetworkWithSda
# import gzip, cPickle
# f = gzip.open("E:\\DIPLOMA\\Eclipse_workspace\\NeuralNets\\deep_learning_examples\\data\\mnist.pkl.gz", 'rb')
# train_set, valid_set, test_set = cPickle.load(f)
# f.close()
# nn = NeuralNetworkWithSda(hidden_levels = [200, 200], in_out = (784, 10), dropout_thresholds = (0.8, 0.5), batch_size = 200, seed = 42, learning_rate = 0.2, n_epochs = 1500,
#                           sda_learning_rate = 0.05, sda_batch_size = 10, sda_corruption_level = 0.1, sda_n_epochs = 15)
# nn.learn(train_set, valid_set, test_set)
# nn.save_model_to_file("E:\\model2.pickle")
# nn.load_model_from_file("E:\\model2.pickle")
# 
# # Test the Theano neural network with SRBM on MNIST
# from impl.neural_network_srbm import NeuralNetworkWithSrbm
# import gzip, cPickle
# f = gzip.open("E:\\DIPLOMA\\Eclipse_workspace\\NeuralNets\\deep_learning_examples\\data\\mnist.pkl.gz", 'rb')
# train_set, valid_set, test_set = cPickle.load(f)
# f.close()
# nn = NeuralNetworkWithSrbm(hidden_levels = [200, 200], in_out = (784, 10), dropout_thresholds = (0.8, 0.5), batch_size = 200, seed = 42, learning_rate = 0.2, n_epochs = 1500,
#                           srbm_learning_rate = 0.05, srbm_batch_size = 200, srbm_k = 1, srbm_persistent = False, srbm_n_epochs = 15)
# nn.learn(train_set, valid_set, test_set)
# nn.save_model_to_file("E:\\model3.pickle")
# nn.load_model_from_file("E:\\model3.pickle")