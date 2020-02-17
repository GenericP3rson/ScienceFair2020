import quantum_test as qnn
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("sine.txt")
inp = data[:, 0]
out = data[:, 1]

qnn1 = qnn.qnn()
qnn1["init"](5)
weights = qnn1["train"](inp, out, 2)
x_pred = np.linspace(-1, 1, 50)
predictions = qnn1["test"](x_pred, weights)

qnn2 = qnn.qnn()
qnn2["set_var"](qnn1["get_var"]())
weights = qnn2["train"](inp, out, 2)
predictions2 = qnn2["test"](x_pred, weights)

plt.figure()
plt.scatter(inp, out)
plt.scatter(x_pred, predictions, color="green")
plt.scatter(x_pred, predictions2, color="red")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tick_params(axis="both", which="major")
plt.tick_params(axis="both", which="minor")
plt.show()
