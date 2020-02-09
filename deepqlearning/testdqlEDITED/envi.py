import numpy as np
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import player
class BlobEnv:
    SIZE = 25 # A 25x25 environment
    RETURN_IMAGES = True # For DQL
    MOVE_PENALTY = -1
    ENEMY_PENALTY = -300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9 # The number of moves
    NUM_OF_PLAYERS = 2
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255),
         4: (255/2, 255/2, 255/2)}
    people = []
    fires = []
    walls = []
    DOORX = 0
    DOORY = SIZE-1

    def check(self, person, arr):
        for i in arr:
            if i == person: 
                print(i, person, i==person)
                return True
        return False

    def createNewPlayer(self, isPlayer= False, item = False):
        '''
        Adds players or objects to the board at a distinct position.
        '''
        if isPlayer:
            person = player.Blob(self.SIZE, True, self.DOORX, self.DOORY)
        else:
            person = player.Blob(self.SIZE)
            while self.check(person, self.people):
                person = player.Blob(self.SIZE)
            if item == "wall":
                self.walls.append(person)
            elif item == "fire":
                self.fires.append(person)
        self.people.append(person)
        return person

    def reset(self):
        self.people = []
        self.walls = []
        self.fires = []
        self.player = self.createNewPlayer(True)
        self.player2 = self.createNewPlayer(True)
        self.food = self.createNewPlayer()
        self.enemy = self.createNewPlayer()
        for i in range(int(self.SIZE/2)):  # Generates Wall
            self.createNewPlayer(False, "wall")
        for i in range(int(self.SIZE/2)):  # Generates Fire
            self.createNewPlayer(False, "fire")
        # Basically, the code above initialises all the players to a distinct section.

        self.episode_step = 0 # Number of episodes we've been through

        # Initial observation
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image()) # obs is the image
        else:
            observation = (self.player-self.food) + (self.player-self.enemy) # obs is their coordinates
        return observation
    # def move(self):
    def step(self, action):
        self.episode_step += 1 # New step
        for i in self.fires:
            if (np.random.randint(0, 50) == 1):  # 2% chance
                new_fire = player.Blob(self.SIZE, True, i.x, i.y)
                self.people.append(new_fire)
                self.fires.append(new_fire)
                i.move()
        self.player.action(action[0], self.walls) # Takes the action (changes the x and y)
        self.player2.action(action[1], self.walls)
        # TODO: Make player2 take an action
        #### MAYBE ###
        # self.enemy.move() # TODO: Make it also take an action
        # self.food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image()) 
            # This will make the observation an image
        else:
            new_observation = (self.player-self.food) + \
                (self.player-self.enemy)
            # This will make the observation part of an array. The indicies will be where we need to go.


        # TODO: Change the reward appropriately.
        if self.player == self.enemy or self.check(self.player, self.fires):
            reward = self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        # elif 
        else:
            reward = self.MOVE_PENALTY

        # NOTE: This is the rewards for player2
        if self.player2 == self.enemy or self.check(self.player2, self.fires):
            reward += self.ENEMY_PENALTY
        elif self.player2 == self.food:
            reward += self.FOOD_REWARD
        # elif
        else:
            reward += self.MOVE_PENALTY

        done = False
        # TODO: Make it end once the guy exits the door? Or once the person is saved?
        if reward >= self.FOOD_REWARD-self.MOVE_PENALTY or reward <= self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True
            # We'll finish after we lost, won, or it's been too long
        self.render() # Shows images
        for i in self.people: print(i)
        print("FIRE: ", self.enemy)
        print("PERSON: ", self.food)
        print("MAN: ", self.player)
        # TODO: Make sure to change this appropriately for 2+ players
        return new_observation, reward, done 
        # We'll return the new board, the reward, and whether we're done or not

    def render(self):
        '''
        Shows the image.
        '''
        img = self.get_image()
        # resizing so we can see our agent in all its glory.
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN 
    def get_image(self):
        '''
        This will set each change to an image.
        Instead of reading numbers, it'll be an image.
        '''
        # starts an rbg of our size
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        # sets the food location tile to green color
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        # sets the enemy location to red
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        for i in self.walls:
            env[i.x][i.y] = self.d[4]
        for i in self.fires:
            env[i.x][i.y] = self.d[self.ENEMY_N]
        # sets the player tile to blue
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
        env[self.player2.x][self.player2.y] = self.d[self.PLAYER_N]
        # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        img = Image.fromarray(env, 'RGB')
        return img
