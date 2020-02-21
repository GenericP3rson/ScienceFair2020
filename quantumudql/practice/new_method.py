'''
We're going to do this half-in-half.
Have a NN encrypt and have the quantum part generate options
'''

# import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import quantum_test as qnn

model = Sequential()

OBSERVATION_SPACE_VALUES = (25, 25, 3)

# OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
model.add(Conv2D(256, (3, 3), input_shape=OBSERVATION_SPACE_VALUES))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# this converts our 3D feature maps to 1D feature vectors
model.add(Flatten())
model.add(Dense(64))

# ACTION_SPACE_SIZE = how many choices (14)
# BUT... 14 moves per x players would be an exponent...
model.add(Dense(17 **2, activation='linear'))
model.summary()
model.compile(loss="mse", optimizer=Adam(
    lr=0.001), metrics=['accuracy'])  # Outputs the probabilities of each move

inp = model.predict(np.array([i for i in range(25*25*3)]).reshape(1, 25, 25, 3)).reshape(289,)
correct = np.array([np.random.randint(1, 10)/10 for i in range(17**2)]).reshape(289,)

qnn1 = qnn.qnn()
qnn1["init"](5)
print(inp.shape)
print(correct.shape)
print(inp, correct)
weights = qnn1["train"](inp, correct, 2)
# print(qnn1["test"](inp[0][0], [weights]))
# print(correct[0][1])

print(qnn1["test"](model.predict(np.array([i for i in range(25*25*3)]).reshape(1, 25, 25, 3)).reshape(289,), 
weights))