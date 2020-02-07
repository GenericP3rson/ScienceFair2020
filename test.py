import torch 
import gym 

env = gym.make('CartPole-v0').unwrapped
print(env.action_space.n)

'''
Two possible actions: left or right.
For us, 10 possible actions: all directions surrounding, spray, and pick up
'''

