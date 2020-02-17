import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

import matplotlib.pyplot as plt

DEVICE = qml.device("strawberryfields.fock", wires=1, cutoff_dim=10)

class QNN():
    def __init__(self, num_of_layers = 4, x = np.array([]), y = np.array([])):
        self.num_of_layers = num_of_layers
        self.inp = x
        self.out = y
    
    def layer(self, v):
        # Matrix multiplication of input layer
        qml.Rotation(v[0], wires=0)
        qml.Squeezing(v[1], 0.0, wires=0)
        qml.Rotation(v[2], wires=0)
        # Bias
        qml.Displacement(v[3], 0.0, wires=0)
        # Element-wise nonlinear transformation
        qml.Kerr(v[4], wires=0)

    def square_loss(self, labels, predictions):
        '''
        Sum of the squares of the differences between the actual and predicted
        '''
        loss = 0
        for actual, predicted in zip(labels, predictions):
            loss += (actual - predicted) ** 2

        loss /= len(labels)
        return loss

    @qml.qnode(DEVICE)
    def quantum_neural_net(self, var = [], x=None):
        # Encode input x into quantum state
        qml.Displacement(x, 0.0, wires=0)
        # "layer" subcircuits
        for v in var: 
            # Matrix multiplication of input layer
            qml.Rotation(v[0], wires=0)
            qml.Squeezing(v[1], 0.0, wires=0)
            qml.Rotation(v[2], wires=0)
            # Bias
            qml.Displacement(v[3], 0.0, wires=0)
            # Element-wise nonlinear transformation
            qml.Kerr(v[4], wires=0)
        return qml.expval(qml.X(0))

    def cost(self, var = [], features = [], labels = []):
        '''
        Grabs the predicted values and finds the loss
        '''
        print(var)
        preds = [self.quantum_neural_net(var, x) for x in features] # Comes with the predictions
        return self.square_loss(labels, preds) # Calculates the loss

    def train(self, iterations = 25):
        opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)
        var_init = 0.05 * np.random.randn(self.num_of_layers, 5) # initialise network weights at a Normal Distribution
        self.weights = var_init # initialise network weights at a Normal Distribution
        for it in range(iterations):
            self.weights = opt.step(lambda v: self.cost(v, X, Y), self.weights)
            print("Iter: {:5d} | Cost: {:0.7f} ".format(it + 1, self.cost(self.weights, X, Y)))
    
    def test(self, inp = np.linspace(-1, 1, 50)):
        return [self.quantum_neural_net(var = self.weights, x=x_) for x_ in inp]

data = np.loadtxt("sine.txt")
X = data[:, 0]
Y = data[:, 1]

i = QNN(4, X, Y)

# plt.figure()
# plt.scatter(X, Y)
# plt.xlabel("x", fontsize=18)
# plt.ylabel("f(x)", fontsize=18)
# plt.tick_params(axis="both", which="major", labelsize=16)
# plt.tick_params(axis="both", which="minor", labelsize=16)
# plt.show()

i.train(25)
x_pred = np.linspace(-1, 1, 50)
predictions = i.test(x_pred)
plt.figure()
plt.scatter(X, Y)
plt.scatter(x_pred, predictions, color="green")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tick_params(axis="both", which="major")
plt.tick_params(axis="both", which="minor")
plt.show()