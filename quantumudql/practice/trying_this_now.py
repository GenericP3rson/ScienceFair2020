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
    Kgate(kappa[i]) | q

    return q


import strawberryfields as sf
from strawberryfields.ops import *

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

fidelity = tf.abs(tf.reduce_sum(tf.conj(ket) * target_state)) ** 2

# Objective function to minimize
cost = tf.abs(tf.reduce_sum(tf.conj(ket) * target_state) - 1)

# Using Adam algorithm for optimization
optimiser = tf.train.AdamOptimizer()
min_cost = optimiser.minimize(cost)

# Begin Tensorflow session
session = tf.Session()
session.run(tf.global_variables_initializer())

fid_progress = []
best_fid = 0

# Run optimization
for i in range(reps):

    # one repitition of the optimization
    _, cost_val, fid_val, ket_val = session.run([min_cost, cost, fidelity, ket])

    # Stores fidelity at each step
    fid_progress.append(fid_val)

    if fid_val > best_fid:
        # store the new best fidelity and best state
        best_fid = fid_val
        learnt_state = ket_val

    # Prints progress at every 10 reps
    if i % 100 == 0:
        print("Rep: {} Cost: {:.4f} Fidelity: {:.4f}".format(i, cost_val, fid_val))


from matplotlib import pyplot as plt
# %matplotlib inline
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
plt.style.use('default')

plt.plot(fid_progress)
plt.ylabel('Fidelity')
plt.xlabel('Step')

import numpy as np 

target_state = np.zeros([cutoff])
target_state[1] = 1