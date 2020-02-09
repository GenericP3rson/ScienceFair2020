import numpy as np
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

class Blob:
    def __init__(self, size, player = False, doorx = 0, doory = 0):
        '''
        Creates a board of size by size.
        '''
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
        # print("TRY", self.x, " ", self.y, type(self.x))
        if player:
            self.x, self.y = \
                np.random.randint(doorx, doorx+1), \
                np.random.randint(doory, doory+1)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        '''
        Subtracts the xs and the ys
        '''
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice, limits):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1, limits=limits)
        elif choice == 1:
            self.move(x=-1, y=-1, limits=limits)
        elif choice == 2:
            self.move(x=-1, y=1, limits=limits)
        elif choice == 3:
            self.move(x=1, y=-1, limits=limits)
        elif choice == 4:
            self.move(x=1, y=0, limits=limits)
        elif choice == 5:
            self.move(x=-1, y=0, limits=limits)
        elif choice == 6:
            self.move(x=0, y=1, limits=limits)
        elif choice == 7:
            self.move(x=0, y=-1, limits=limits)
        elif choice == 8:
            self.move(x=0, y=0, limits=limits)

    def move(self, x=False, y=False, limits = []):
        '''
        This adds either the correct amount or a random amount depending on the parameters.
        '''
        for i in limits:
            print(i.x == self.x and i.y == self.y)
            if i.x == self.x and i.y == self.y:
                x = 0
                y = 0
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

        # TODO: Fix if on person or touching furniture