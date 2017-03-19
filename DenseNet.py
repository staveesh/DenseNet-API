"""
Collaborators : Taveesh Sharma, Kushagr Arora, Vishal Agrawal

Gates Implemented:
	ReLU
	Sigmoid
	Linear
	Softmax

Loss Functions:
	L1
	L2
	Cross Entropy
	SVM
	
Optimisers:
	SGD
	SGD with Momentum
"""
import numpy as np
from Graph_API import Graph,Cross_Entropy,L1,L2,SVM,Linear,ReLU,Sigmoid,Softmax,Optimiser

class DenseNet:
    def __init__(self, input_dim, optim_config, loss_fn):

        self.graph = Graph(input_dim,optim_config,loss_fn)
        
    def addlayer(self, activation, units):

        self.graph.addgate(activation,units)

    def train(self, X, Y):
		# Choose i at random to send 1 example at a time
        i = np.random.randint(0,len(X))
        x = np.atleast_2d(X[i])
        y = Y[i]
        predicted = self.graph.forward(x)
        loss_value = self.graph.backward(y)
        self.graph.update()
        return loss_value

    def predict(self, X):
        
        return self.graph.forward(X)
