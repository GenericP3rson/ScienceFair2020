import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

prog = sf.Program(2)

phi = tf.placeholder(tf.float32)
alpha = tf.Variable(0.5)

with prog.context as q:
    Dgate(alpha) | q[0]
    MeasureHomodyne(phi) | q[0]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {phi: 0.0}

eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})
results = eng.run(prog, run_options={"eval": False})

state_density_matrix = results.state.dm()
homodyne_meas = results.samples[0]
dm_x, meas_x = sess.run(
    [state_density_matrix, homodyne_meas], feed_dict={phi: 0.0})


input_ = tf.placeholder(tf.float32, shape=(2, 1))
weights = tf.Variable([[0.1, 0.1]])
bias = tf.Variable(0.0)
NN = tf.sigmoid(tf.matmul(weights, input_) + bias)
NNDgate = Dgate(NN)


# @sf.convert
def sigmoid(x):
    return tf.sigmoid(x)


prog = sf.Program(2)

with prog.context as q:
    MeasureX | q[0]
    Dgate(sf.convert(sigmoid(q[0]))) | q[1]
