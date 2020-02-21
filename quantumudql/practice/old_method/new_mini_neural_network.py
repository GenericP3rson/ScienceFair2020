import strawberryfields as sf
from strawberryfields.ops import *

prog = sf.Program(2)

import tensorflow as tf

input_ = tf.placeholder(tf.float32, shape=(2, 1))
weights = tf.Variable([[0.1, 0.1]])
bias = tf.Variable(0.0)
NN = tf.sigmoid(tf.matmul(weights, input_) + bias)
NNDgate = Dgate(NN)


@sf.convert
def sigmoid(x):
    return tf.sigmoid(x)


prog = sf.Program(2)

with prog.context as q:
    MeasureX | q[0]
    Dgate(sigmoid(q[0])) | q[1]

batch_size = 3
prog = sf.Program(2)
eng = sf.Engine('tf', backend_options={
                "cutoff_dim": 7, "batch_size": batch_size})

with prog.context as q:
    Dgate(tf.Variable([0.1] * batch_size)) | q[0]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {input_: [[0.0], [0.0]]}
result = eng.run(prog, run_options={"session": sess, "feed_dict": feed_dict})

print("DONE")
print(result)
