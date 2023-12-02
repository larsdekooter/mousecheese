import pygame
import time
from data import cooldown

class Mouse:
    def __init__(self, x, y):
        self.img = pygame.transform.scale(pygame.image.load("mouse.png"), (100, 100))
        self.x = x
        self.y = y
        self.cd = False
        self.lastCooldown = None
        self.moves = 0
    
    def move(self, move):
        # [0 = UP,0 = DOWN,0 = LEFT,0 = RIGHT]
        if self.cd:
            if time.time() - self.lastCooldown >=cooldown:
                self.cd = False
            else:
                return
        if move[0] == 1 and self.y != 0:
            self.y -= 100
            self.lastCooldown = time.time()
            self.cd = True
            self.moves +=1
        elif move[1] == 1 and self.y != 700:
            self.y += 100
            self.lastCooldown = time.time()
            self.cd = True
            self.moves +=1
        elif move[2] == 1 and self.x != 0:
            self.x -=100
            self.lastCooldown = time.time()
            self.cd = True
            self.moves +=1
        elif move[3] == 1 and self.x != 700:
            self.x += 100
            self.lastCooldown = time.time()
            self.cd = True
            self.moves +=1
        else:
            self.cd = False
            self.lastCooldown = None