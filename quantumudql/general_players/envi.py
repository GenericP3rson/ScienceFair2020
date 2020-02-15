import numpy as np
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import player


class BlobEnv:
    SIZE = 25  # A 25x25 environment
    RETURN_IMAGES = True  # For DQL
    MOVE_PENALTY = -1
    ENEMY_PENALTY = -300
    FOOD_REWARD = 200
    KILL_FIRE_REWARD = 10
    ALL_FIRES_KILLED = 100
    RESOURCE_USE_PENALTY = -10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 14  # The number of moves
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (203, 192, 255),
         2: (0, 255, 0),
         3: (0, 0, 255),
         4: (255/2, 255/2, 255/2),
         5: (255, 175, 0)}
    people = []
    fires = []
    walls = []
    DOORX = 0
    DOORY = SIZE-1
    shot = [False]

    def __init__(self, num):
        self.NUM_OF_PLAYERS = num

    def check(self, person, arr, begone=False):
        for i in arr:
            if i == person:
                # print(i, person, i==person)
                # if begone:
                #     self.players.remove(i)
                #     self.people.remove(i)
                return True
        return False

    def createNewPlayer(self, isPlayer=False, item=False):
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
        self.people = []  # All entities; used to make sure nothing collides
        self.walls = []
        self.fires = []
        self.players = []
        for i in range(self.NUM_OF_PLAYERS):  # Creates x number of players
            self.players.append(self.createNewPlayer(True))
        # self.player = self.createNewPlayer(True)
        # self.player2 = self.createNewPlayer(True)
        self.food = self.createNewPlayer()
        # self.enemy = self.createNewPlayer()
        for i in range(int(self.SIZE/2)):  # Generates Wall
            self.createNewPlayer(False, "wall")
        for i in range(int(self.SIZE/2)):  # Generates Fire
            self.createNewPlayer(False, "fire")
        # Basically, the code above initialises all the players to a distinct section.

        self.episode_step = 0  # Number of episodes we've been through

        # Initial observation
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())  # obs is the image
        # else:
        #     observation = (self.player-self.food) + (self.player-self.enemy) # obs is their coordinates
        return observation
    # def move(self):

    def killFire(self, x=False, y=False):
        if ((not x and not y) or len(self.fires) == 0 or x < 0 or y < 0 or x >= self.SIZE or y >= self.SIZE):
            return self.RESOURCE_USE_PENALTY
        self.shot = [x, y]
        for i in (self.fires):
            if x == i.x and y == i.y:
                self.fires.remove(i)
                self.people.remove(i)
                if len(self.fires) == 0:  # If all fires now killed, extra reward
                    return self.ALL_FIRES_KILLED
                return self.KILL_FIRE_REWARD  # Else just a reward for killing the fire
        return self.RESOURCE_USE_PENALTY

    def step(self, action):
        self.shot = [False]
        self.episode_step += 1  # New step
        for i in self.fires:
            if (np.random.randint(0, 50) == 1):  # 2% chance
                new_fire = player.Blob(self.SIZE, True, i.x, i.y)
                self.people.append(new_fire)
                self.fires.append(new_fire)
                i.move()
        # rew1a, rew1b = self.player.action(action[0], self.walls) # Takes the action (changes the x and y)
        # rew2a, rew2b = self.player2.action(action[1], self.walls)
        # TODO: Generalise actions
        #### MAYBE ###
        # self.enemy.move() # TODO: Make it also take an action
        # self.food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
            # This will make the observation an image
        # else:
        #     new_observation = (self.player-self.food) + \
        #         (self.player-self.enemy)
            # This will make the observation part of an array. The indicies will be where we need to go.

        # TODO: Change the reward appropriately.
        reward = 0
        # print("Let's Go", len(self.players), len(action))
        # print("DOES THIS WORK 1")
        players_to_remove = []
        for i in range(len(self.players)):
            reward1, reward2 = self.players[i].action(
                action[i], self.walls)  # Take action
            # print("DOES THIS WORK 2")
            if reward1 and reward2:  # If it shoots water
                reward += self.killFire(reward1, reward2)
            if self.check(self.players[i], self.fires, True):
                reward += self.ENEMY_PENALTY  # Checks if it is touching fire
                players_to_remove.append(i)
                # TODO: Kill the player
            elif self.players[i] == self.food:
                reward += self.FOOD_REWARD  # If it has found child
            else:
                reward += self.MOVE_PENALTY  # Else, just move
        # print("DOES THIS WORK 3")actions = convert_num_to_action_array(unique_action_index)
        for i in players_to_remove[::-1]:
            self.players.pop(i)
        print("Reward: ", reward)
        done = False
        # TODO: Make it end once the guy exits the door? Or once the person is saved?
        if self.check(self.food, self.players) or len(self.players) == 0 or self.episode_step >= 200:
            done = True
            # We'll finish after we lost, won, or it's been too long
        self.render()  # Shows images
        # for i in self.people: print(i)
        # print("FIRE: ", self.enemy)
        # print("PERSON: ", self.food)
        # print("MAN: ", self.player)
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
        # env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        for i in self.walls:
            env[i.x][i.y] = self.d[4]
        for i in self.fires:
            env[i.x][i.y] = self.d[self.ENEMY_N]
        if self.shot[0]:
            env[self.shot[0]][self.shot[1]] = self.d[5]
        # sets the player tile to pink
        for i in self.players:
            env[i.x][i.y] = self.d[self.PLAYER_N]
        img = Image.fromarray(env, 'RGB')
        return img
