import numpy as np 

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

class Graph:
    
    def __init__(self, input_dim, optim_config, loss_fn):
        self.gates = []    
        self.input_dim = input_dim
        self.optim_config = optim_config
        self.loss_fn = loss_fn
        self.weights = []
        self.layers = [input_dim[1]]
        self.deltas = None
        self.fwd = None

    def addgate(self, activation, units=0): 
        if activation.lower() == 'relu':
            self.gates.append(ReLU(units,self.weights,self.layers))
        elif activation.lower() == 'sigmoid':
            self.gates.append(Sigmoid(units,self.weights,self.layers))
        elif activation.lower() == 'softmax':
            self.gates.append(Softmax(units,self.weights,self.layers))
        elif activation.lower() == 'linear':
            self.gates.append(Linear(self.units,self.weights,self.layers))

    def forward(self, input):
        self.fwd = []
        signal = input
        for i in range(len(self.gates)):
            signal = self.gates[i].forward(signal)
            self.fwd.append((self.gates[i].input,signal))
        return signal

    def backward(self, expected):
        _loss = None
        predicted = self.fwd[-1][1]
        self.deltas = [None for i in range(len(self.layers)-1)]
        if self.loss_fn.lower() == 'l1':
            _loss = L1(expected,predicted)
        elif self.loss_fn.lower() == 'l2':
            _loss = L2(expected,predicted)
        elif self.loss_fn.lower() == 'cross entropy':
            _loss = Cross_Entropy(expected,predicted)
        elif self.loss_fn.lower() == 'svm':
            _loss = SVM(expected,predicted)

        dz = _loss.grad()
        for i in reversed(range(len(self.layers)-1)):
            self.deltas[i] = self.gates[i].backward(dz)
            if i != 0:
                dz = self.gates[i].lin.backward(self.deltas[i])
        return _loss.loss()
        
    def update(self):
        ls = []
        for i in range(len(self.weights)):
            w_before = self.weights[i][0]
            b_before = self.weights[i][1]
            dW = self.optim_config.learning_rate*np.dot(self.fwd[i][0].T,self.deltas[i])
            db = self.optim_config.learning_rate*self.deltas[i]
            if len(self.optim_config.weights) == 0:
                ls.append((dW,db))
            else:
                dW += self.optim_config.momentum_eta*self.optim_config.weights[-1][i][0]
                db += self.optim_config.momentum_eta*self.optim_config.weights[-1][i][1]
                ls.append((dW,db))
            w_after = w_before - dW
            b_after = b_before - db
            self.weights[i] = (w_after,b_after)
        self.optim_config.weights.append(ls)

class ReLU:

    def __init__(self,units,weights,layers):
        self.input = None
        self.local_grad = None
        self.lin = Linear(units,weights,layers)

    def forward(self, x):
        self.input = x
        x = self.lin.forward(x)
        self.local_grad = (x>0)
        return x*self.local_grad

    def backward(self, dz):
        res = self.local_grad*dz
        dz = self.lin.backward(res)
        return res 

class Sigmoid:

    def __init__(self,units,weights,layers):
        self.input = None
        self.local_grad = None
        self.lin = Linear(units,weights,layers)

    def forward(self, x):
        self.input = x
        x = self.lin.forward(x)
        f = 1.0/(1.0+np.exp(-x))
        self.local_grad = f*(1-f)
        return f

    def backward(self, dz):
        res = self.local_grad*dz
        return res  

class Softmax:

    def __init__(self,units,weights,layers):
        self.input = None
        self.local_grad = None
        self.lin = Linear(units,weights,layers)

    def forward(self,x):
        self.input = x
        x = self.lin.forward(x)
        f = np.exp(x)/np.sum(np.exp(x))
        l = len(f)
        self.local_grad = np.zeros(f.shape)
        for i in range(l):
            for j in range(l):
                if i == j:
                    self.local_grad[i][j] = f[0][i]*(1-f[0][i])
                else:
                    self.local_grad[i][j] = -1.0*f[0][i]*f[0][j]
        return f

    def backward(self,dz):
        res = self.local_grad*dz
        return res 

class Linear:
    
    def __init__(self,units,weights,layers):
        self.input = None       
        self.w = 2*np.random.random((layers[-1],units))-1
        self.b = 2*np.random.random((1,units))-1
        weights.append((self.w,self.b))
        layers.append(units)

    def forward(self,x):
        self.input = x
        return np.dot(self.input,self.w)+self.b

    def backward(self,dz):
        return np.dot(dz,self.w.T)

class L2:
    def __init__(self,expected,predicted):
        self.expected = expected
        self.predicted = predicted

    def loss(self):
        return np.linalg.norm(self.expected-self.predicted)/self.expected.shape[0]

    def grad(self):
        return self.predicted-self.expected

class L1:   
    def __init__(self,expected,predicted):
        self.expected = expected
        self.predicted = predicted
        self.diff = None

    def loss(self):
        self.diff = self.expected-self.predicted
        return np.sum(np.abs(self.diff))/self.expected.shape[0]

    def grad(self):
        return -1.0*(self.diff > 0) + 1.0*(self.diff < 0)

class Cross_Entropy:

    def __init__(self,expected,predicted):
        self.expected = expected
        self.predicted = predicted

    def loss(self):
        return -np.sum(self.expected*np.log(self.predicted))

    def grad(self):
        return -self.expected/self.predicted

class SVM:

    def __init__(self,expected,predicted):
        self.expected = expected
        self.predicted = predicted
        self.out = np.zeros(len(expected))

    def loss(self):
        l = 0.0
        ind = np.argmax(self.expected,axis=0)
        for i in range(len(self.predicted)):
            if i == ind:
                continue
            self.out = self.predicted[0][i]-self.predicted[0][ind]+1
            l += np.maximum(0,self.out)
        return l

    def grad(self):
        return 1.0*(self.out > 0)

class Optimiser:

    def __init__(self, learning_rate, momentum_eta = 0.0):
        self.weights = []
        self.learning_rate = learning_rate
        self.momentum_eta = momentum_eta
