#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import *

class CV_QNN():
    def __init__(self):
        # define width and depth of CV quantum neural network
        self.modes = 1
        self.layers = 25  # Number of layers
        self.cutoff_dim = 6  # Max number of nodes per layer

        # defining desired state (single photon state)
        target_state = np.zeros(self.cutoff_dim)
        target_state[1] = 1
        self.target_state = tf.constant(target_state, dtype=tf.complex64)  # Desired output
        self.starting_stuff()

    def setup(self):
        N = len(self.layers)
        BS_variable_number = int(N * (N - 1) / 2)
        R_variable_number = max(1, N - 1)

        theta_variables_1 = tf.Variable(
            tf.random_normal(shape=[BS_variable_number]))
        phi_variables_1 = tf.Variable(
            tf.random_normal(shape=[BS_variable_number]))
        rphi_variables_1 = tf.Variable(
            tf.random_normal(shape=[R_variable_number]))

        theta_variables_2 = tf.Variable(
            tf.random_normal(shape=[BS_variable_number]))
        phi_variables_2 = tf.Variable(
            tf.random_normal(shape=[BS_variable_number]))
        rphi_variables_2 = tf.Variable(
            tf.random_normal(shape=[R_variable_number]))

        s_variables = tf.Variable(tf.random_normal(shape=[N], stddev=0.0001))
        d_variables_r = tf.Variable(tf.random_normal(shape=[N], stddev=0.0001))
        d_variables_phi = tf.Variable(tf.random_normal(shape=[N]))
        k_variables = tf.Variable(tf.random_normal(shape=[N], stddev=0.0001))
    
    def starting_stuff(self):
        # initialize engine and program objects
        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        qnn = sf.Program(self.modes)


        self.modified_inputs = tf.placeholder(tf.float32, shape=[self.layers])

        with qnn.context as q:
            for _ in range(self.layers):
                self.layer(q)

        # starting the engine
        results = eng.run(qnn, run_options={"eval": False})
        ket = results.state.ket()

        # defining cost function
        difference = tf.reduce_sum(tf.abs(ket - self.target_state))
        fidelity = tf.abs(tf.reduce_sum(tf.conj(ket) * self.target_state)) ** 2
        cost = difference

        # setting up optimizer
        optimiser = tf.train.AdamOptimizer()
        minimize_cost = optimiser.minimize(cost)

        print("Beginning optimization")

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        x = np.array([i for i in range(25)])/25

        feed_dict = {self.modified_inputs: x}

        fidelity_before = sess.run(fidelity, feed_dict=feed_dict)

        for i in range(10):
            sess.run(minimize_cost, feed_dict=feed_dict)
            print(str(i) + " / 10")

        fidelity_after = sess.run(fidelity, feed_dict=feed_dict)

        print("Fidelity before optimization: " + str(fidelity_before))
        print("Fidelity after optimization: " + str(fidelity_after))
        print("\nTarget state: " + str(sess.run(self.target_state, feed_dict=feed_dict)))
        print("Output state: " + str(np.round(sess.run(ket, feed_dict=feed_dict), decimals=3)))

        qnn.draw_circuit()


    # define interferometer
    def interferometer(self, theta, phi, rphi, q):
        """Parameterised interferometer acting on N qumodes
        Args:
            theta (list): list of length N(N-1)/2 real parameters
            phi (list): list of length N(N-1)/2 real parameters
            rphi (list): list of length N-1 real parameters
            q (list): list of qumodes the interferometer is to be applied to
        """
        N = len(q)

        if N == 1:
            # the interferometer is a single rotation
            Rgate(rphi[0]) | q[0]
            return

        n = 0  # keep track of free parameters

        # Apply the rectangular beamsplitter array
        # The array depth is N
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                # skip even or odd pairs depending on layer
                if (l + k) % 2 != 1:
                    BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1

        # apply the final local phase shifts to all modes except the last one
        for i in range(max(1, N - 1)):
            Rgate(rphi[i]) | q[i]
        # Rgate only applied to first N - 1 modes
    def read_all(self):
        self.variables = [self.theta_variables_1, self.phi_variables_1, self.rphi_variables_1, self.theta_variables_2, 
                    self.phi_variables_2, self.rphi_variables_2, self.s_variables, self.d_variables_r,
                    self.d_variables_phi, self.k_variables]
        for i in range(len(self.variables)):
            self.variables[i] = self.variables[i].read_value()
        print(self.variables)
        return self.variables
    # define layer
    def layer(self, q):
        """CV quantum neural network layer acting on N modes
        Args:
            q (list): list of qumodes the layer is to be applied to
        """
        N = len(q)
        BS_variable_number = int(N * (N - 1) / 2)
        R_variable_number = max(1, N - 1)
        # Creating initial starting points

        for i in range(N):
            # print(modified_inputs[i])
            Rgate(self.modified_inputs[i]) | q[i]

        self.theta_variables_1 = tf.Variable(
            tf.random_normal(shape=[BS_variable_number]))
        self.phi_variables_1 = tf.Variable(
            tf.random_normal(shape=[BS_variable_number]))
        self.rphi_variables_1 = tf.Variable(
            tf.random_normal(shape=[R_variable_number]))

        self.theta_variables_2 = tf.Variable(
            tf.random_normal(shape=[BS_variable_number]))
        self.phi_variables_2 = tf.Variable(
            tf.random_normal(shape=[BS_variable_number]))
        self.rphi_variables_2 = tf.Variable(
            tf.random_normal(shape=[R_variable_number]))

        self.s_variables = tf.Variable(
            tf.random_normal(shape=[N], stddev=0.0001))
        self.d_variables_r = tf.Variable(
            tf.random_normal(shape=[N], stddev=0.0001))
        self.d_variables_phi = tf.Variable(tf.random_normal(shape=[N]))
        self.k_variables = tf.Variable(
            tf.random_normal(shape=[N], stddev=0.0001))

        # begin layer
        self.interferometer(self.theta_variables_1,
                            self.phi_variables_1, self.rphi_variables_1, q)

        for i in range(N):
            Sgate(self.s_variables[i]) | q[i]

        self.interferometer(self.theta_variables_2,
                            self.phi_variables_2, self.rphi_variables_2, q)

        for i in range(N):
            Dgate(self.d_variables_r[i], self.d_variables_phi[i]) | q[i]
            Kgate(self.k_variables[i]) | q[i]
        # end layer
        self.read_all()

i = CV_QNN()
