import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=10)

def qnn():
    NUM_OF_LAYERS = 4
    var = 0.05 * np.random.randn(NUM_OF_LAYERS, 5)

    def set_constants(num_of_layers = NUM_OF_LAYERS):
        NUM_OF_LAYERS = num_of_layers
        var = 0.05 * np.random.randn(NUM_OF_LAYERS, 5)

    def set_var(new_var):
        var = new_var

    def get_var():
        return var

    def layer(v):
        # Matrix multiplication of input layer
        qml.Rotation(v[0], wires=0)
        qml.Squeezing(v[1], 0.0, wires=0)
        qml.Rotation(v[2], wires=0)

        # Bias
        qml.Displacement(v[3], 0.0, wires=0)

        # Element-wise nonlinear transformation
        qml.Kerr(v[4], wires=0)


    @qml.qnode(dev)
    def quantum_neural_net(var, x=None):
        # Encode input x into quantum state
        qml.Displacement(x, 0.0, wires=0)

        # "layer" subcircuits
        for v in var:
            layer(v)

        return qml.expval(qml.X(0))

    def square_loss(labels, predictions):
        loss = 0
        for actual, predicted in zip(labels, predictions):
            loss += (actual - predicted) ** 2

        loss /= len(labels)
        return loss

    def cost(var, features, labels):
        # print(var)
        preds = [quantum_neural_net(var, x=x) for x in features] # Comes with the predictions
        return square_loss(labels, preds) # Calculates the loss

    def weight_init(num_of_layers = NUM_OF_LAYERS):
        # np.random.seed(0)
        var_init = 0.05 * np.random.randn(num_of_layers, 5) # initialise network weights at a Normal Distribution
        # four by five layer network
        # print(var_init)
        return var_init

    def train(X =[], Y=[],iterations = 25, num_of_layers = NUM_OF_LAYERS):
        opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

        var = weight_init(num_of_layers)
        for it in range(iterations):
            var = opt.step(lambda v: cost(v, X, Y), var)
            print("Iter: {:5d} | Cost: {:0.7f} ".format(it + 1, cost(var, X, Y)))
        return var

    def test(X, var):
        return [quantum_neural_net(var, x=x_) for x_ in X]

    return {"test": test, "train":train, "init": set_constants, "set_var": set_var, "get_var": get_var}
# if __name__ == "__main__":
     

#     '''
#     DEFINES THE DATA
#     '''
#     data = np.loadtxt("sine.txt")
#     X = data[:, 0]
#     Y = data[:, 1]

#     plt.figure()
#     plt.scatter(X, Y)
#     plt.xlabel("x", fontsize=18)
#     plt.ylabel("f(x)", fontsize=18)
#     plt.tick_params(axis="both", which="major", labelsize=16)
#     plt.tick_params(axis="both", which="minor", labelsize=16)
#     plt.show()

#     '''
#     Initialise the weights
#     '''
#     var = train(NUM_OF_LAYERS, 5)
#     x_pred = np.linspace(-1, 1, 50)
#     predictions = test(x_pred, var)
    

#     plt.figure()
#     plt.scatter(X, Y)
#     plt.scatter(x_pred, predictions, color="green")
#     plt.xlabel("x")
#     plt.ylabel("f(x)")
#     plt.tick_params(axis="both", which="major")
#     plt.tick_params(axis="both", which="minor")
#     plt.show()

#     variance = 1.0

#     plt.figure()
#     x_pred = np.linspace(-2, 2, 50)
#     for i in range(7):
#         rnd_var = variance * np.random.randn(NUM_OF_LAYERS, 7)
#         predictions = [quantum_neural_net(rnd_var, x=x_) for x_ in x_pred]
#         plt.plot(x_pred, predictions, color="black")
#     plt.xlabel("x")
#     plt.ylabel("f(x)")
#     plt.tick_params(axis="both", which="major")
#     plt.tick_params(axis="both", which="minor")
#     plt.show()
