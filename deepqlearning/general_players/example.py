import numpy as np
import keras.backend.tensorflow_backend as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import mtf
import envi
import player

NUM_OF_PLAYERS = 2

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
# Minimum number of steps in a memory to start training
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 10 # Number of iterations

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

past_min_reward = -10000

env = envi.BlobEnv(NUM_OF_PLAYERS)

# For stats
ep_rewards = [-200]

# For more repetitive results
# random.seed(1)
# np.random.seed(1)
# tf.set_random_seed(1)

# Memory fraction, used mostly when training multiple agents
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) 
        # Deque: Can add/remove from front and back
        # print("HMMM", self.replay_memory)

        # Custom tensorboard object
        self.tensorboard = mtf.ModifiedTensorBoard(
            log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
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
        model.add(Dense(env.ACTION_SPACE_SIZE**env.NUM_OF_PLAYERS, activation='linear'))
        model.summary()
        model.compile(loss="mse", optimizer=Adam(
            lr=0.001), metrics=['accuracy']) # Outputs the probabilities of each move
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE: # If not right size
            return
        # print("ARE WE TRAINING???")
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # Let's take a random sample of what we've already seen

        # print("MINI", minibatch)
        # TODO?: Let's see if we can get the last few plus a random few

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0]
                                   for transition in minibatch])/255 # Image
        # print(current_states)
        # This will be for ALL q values
        current_qs_list = self.model.predict(current_states)#[:env.ACTION_SPACE_SIZE]
        # current_qs_list2 = self.model.predict(current_states)[env.ACTION_SPACE_SIZE:]

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array(
            [transition[3] for transition in minibatch])/255 # New Image
        future_qs_list = self.target_model.predict(new_current_states)#[:env.ACTION_SPACE_SIZE]
        # future_qs_list2 = self.target_model.predict(new_current_states)[env.ACTION_SPACE_SIZE:]
        # TODO: Finish making it work with several
        # TODO: After that, generalise

        X = []
        y = []

        # print(minibatch[0])

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # print(index, current_state, action, reward, new_current_state, done)
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            # TODO: Change max to average of maxes?
            if not done:
                # print("WHERE 1")
                max_future_q = np.max(future_qs_list[index])
                # print("WHERE 2")
                new_q = reward + DISCOUNT * max_future_q
                # maxes = 0
                # i = env.ACTION_SPACE_SIZE
                # print("ERROR1")
                # print(future_qs_list[index])
                # print("ERROR2")
                # print(
                #     # future_qs_list[index][i-env.ACTION_SPACE_SIZE:env.ACTION_SPACE_SIZE])
                # print("NO ERROR")
                # for i in range(env.ACTION_SPACE_SIZE, len(future_qs_list[index])+1, env.ACTION_SPACE_SIZE):
                #     maxes+=np.max(future_qs_list[index][i-env.ACTION_SPACE_SIZE:i])
                # print("WHERE 3")
                # new_q = reward + DISCOUNT * maxes
                # print("WHERE 4")
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            # print(current_qs[index].shape, "HELLO", len(current_qs[index]))
            # current_qs[action] = new_q
            # TODO: TRANSLATION
            # We have x actions and 14^x slots
            unique_action_index = 0
            for i in range(len(actions)):
                unique_action_index += actions[-i-1]*env.ACTION_SPACE_SIZE**i 
                # Each gets its own little slot
            current_qs[unique_action_index] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


agent = DQNAgent()


def convert_num_to_action_array(unique_action_index):
    # TODO: CHECK THIS OUT
    action_array = []
    # Left to right?
    exponent = env.NUM_OF_PLAYERS-1
    index = unique_action_index
    while index != 0:
        temporary_number = 0
        while env.ACTION_SPACE_SIZE**exponent <= index:
            # print(env.ACTION_SPACE_SIZE**exponent, index)
            temporary_number += 1
            index -= env.ACTION_SPACE_SIZE**exponent
        exponent -= 1
        action_array.append(temporary_number)
    while len(action_array) < env.NUM_OF_PLAYERS:
        action_array = [0] + action_array
    return action_array



# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        actions = []
        if np.random.random() > epsilon:
            # Get action from Q table
            # NOTE: RETURN HERE
            # TODO: Change to getting two actions
            current_qs = agent.get_qs(current_state)
            unique_action_index = np.argmax(current_qs) # Getting the max value of action combinations
            print(unique_action_index)
            actions = convert_num_to_action_array(unique_action_index)
            # for i in range(env.ACTION_SPACE_SIZE, len(current_qs)+1, env.ACTION_SPACE_SIZE):
            #     actions.append(np.argmax(current_qs[i-env.ACTION_SPACE_SIZE:i]))
            #     print("MAX", np.argmax(current_qs[i-env.ACTION_SPACE_SIZE:i]))
                # action2 = np.argmax(agent.get_qs(current_state)[env.ACTION_SPACE_SIZE:])
            # action2 = np.argmax(agent.get_qs(current_state)[env.ACTION_SPACE_SIZE:])
            # print("")
        else:
            # Get random action
            # TODO: Two random actions
            # action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            # action2 = np.random.randint(0, env.ACTION_SPACE_SIZE)
            current_qs = agent.get_qs(current_state)
            unique_action_index = np.random.randint(0, env.ACTION_SPACE_SIZE**env.NUM_OF_PLAYERS)
            actions = convert_num_to_action_array(unique_action_index)
            # for i in range(env.ACTION_SPACE_SIZE, len(current_qs)+1, env.ACTION_SPACE_SIZE):
            #     actions.append(np.random.randint(0, env.ACTION_SPACE_SIZE))
        # Here's how we're going to do it:
        # 00, 01, 02, 03, ..., 09, 10, 11, 12, ..., 19, 20, 21, ..., 99
        # For the NN / q array
        new_state, reward, done = env.step(actions)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory(
            (current_state, np.array(actions), reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(
            ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(
            reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= past_min_reward:
            past_min_reward = min_reward
            agent.model.save(
                f'models/{NUM_OF_PLAYERS}_players/{min_reward}/{MODEL_NAME}__{env.SIZE}__{NUM_OF_PLAYERS}players__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
agent.model.save(
    f'models/{NUM_OF_PLAYERS}_players/{NUM_OF_PLAYERS}_FINAL_MODEL__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
