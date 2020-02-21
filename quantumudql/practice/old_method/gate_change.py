
import strawberryfields as sf
from strawberryfields.ops import *

# Cutoff dimension
cutoff = 10

# Number of layers
depth = 20

# Number of steps in optimization routine performing gradient descent
reps = 1000

# Standard deviation of initial parameters
passive_sd = 0.1
active_sd = 0.001

import tensorflow as tf

# squeeze gate
sq_r = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))
sq_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))

# displacement gate
d_r = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))
d_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))

# rotation gates
r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))
r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=passive_sd))

# kerr gate
kappa = tf.Variable(tf.random_normal(shape=[depth], stddev=active_sd))

params = [r1, sq_r, sq_phi, r2, d_r, d_phi, kappa]

# layer architecture
def layer(i, q):
    Rgate(r1[i]) | q
    Sgate(sq_r[i], sq_phi[i]) | q
    Rgate(r2[i]) | q
    Dgate(d_r[i], d_phi[i]) | q
    Dgate(d_r[i]) | q
    Kgate(kappa[i]) | q

    return q




# Create program and engine
prog = sf.Program(1)
eng = sf.Engine('tf', backend_options={"cutoff_dim": cutoff})

# Apply circuit of layers with corresponding depth
with prog.context as q:
    for k in range(depth):
        layer(k, q[0])

# Run engine
state = eng.run(prog, run_options={"eval": False}).state
ket = state.ket()


