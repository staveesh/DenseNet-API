# DenseNet-API
A simple Artificial Neural Network API for Supervised Learning using Python.

Collaborators : Taveesh Sharma, Kushagr Arora, Vishal Agrawal

The API currently supports:

##### 1. Activation functions:

  * ReLU
  * Sigmoid
  * Softmax
  * Linear

##### 2. Loss functions:

  * L1 loss
  * L2 loss
  * Cross Entropy
  * SVM loss

##### 3. Optimisers:

  * SGD 
  * SGD with momentum

### Using the API:

```python
from DenseNet import DenseNet
from Graph_API import Optimiser

opti = Optimiser(learning_rate=0.09, momentum_eta=0.5)

X = # Inputs
y = # Labels
net = DenseNet(X.shape,opti,'cross entropy') # 'l1' for L1 loss, 'l2' for L2 loss, 'svm' for svm

# To add a hidden/output layer, simply call addlayer function with activation function and number of neurons.

net.addlayer(activation='sigmoid',units=4)
net.addlayer(activation='relu', units=3)

# For output layer, add units same as the number of classes

net.addlayer(activation='softmax',units=y.shape[1])
```

Note : The train function runs SGD for 1 iteration only. Call it for multiple iterations for training.

```python 
iterations = 10000
for i in range(iterations):
  error = net.train(X,y)
```
