# import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *

eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})

with eng:
    Coherent(1+0.5j)| q[0]
    Squeezed(-2) | q[1]
    Squeezed(2) | q[2]

    # BS = BS

state = eng.run("fock", cutoff_dim = 10)

# prog = sf.Program(2)

# alpha = tf.Variable(0.5)
# theta_bs = tf.constant(0.0)
# phi_bs = tf.sigmoid(0.0)  # this will be a tf.Tensor object
# phi = tf.placeholder(tf.float32)

# with prog.context as q:
#     # States
#     Coherent(alpha) | q[0]

#     # Gates
#     BSgate(theta_bs, phi_bs) | (q[0], q[1])

#     # Measurements
#     MeasureHomodyne(phi) | q[0]

# eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 7})

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# feed_dict = {phi: 0.0}

# results = eng.run(prog, run_options={"session": sess, "feed_dict": feed_dict})

