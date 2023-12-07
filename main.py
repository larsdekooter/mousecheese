from game import Game
import pygame
from time import time
from network import Network
from data import testLength
import data
import sys
import traceback
from datetime import datetime
import numpy as np

def err(type, value, tb):
    print("Execution date time: {}".format(datetime.now()))
    print(traceback.print_tb(tb))
    print(type, ": ", value)

sys.excepthook = err

def train():
    network = Network()
    game = Game()
    while network.nGames < testLength + 100:
        epsilon = data.minEpsilon + (data.maxEpsilon - data.minEpsilon) * np.exp(-data.decayRate * network.decayStep)
        state = network.getState(game)
        move = network.getAction(state)

        done, reward, won = game.step(move)
        stateNew = network.getState(game)

        network.trainShortMemory(state, move, reward, stateNew, done)
        network.remember(state, move, reward, stateNew, done)
        if network.nGames > testLength:
            print(reward)

        if done:
            moves = network.moves
            network.moves = [0,0,0,0]
            totalMoves = 0
            for x in moves:
                totalMoves += x
            if totalMoves == 0:
                totalMoves = 1
            gameTime = time() - game.gameTime
            randomSteps, aiSteps = network.randomSteps, network.aiSteps
            network.aiSteps = 0
            network.randomSteps = 0
            x,y = game.mouse.x, game.mouse.y
            game.reset()
            network.nGames+=1
            network.trainLongMemory()

            print("game", network.nGames, "won", won, "x", x, "y", y, '%', round(aiSteps / (aiSteps+randomSteps) * 100, 2), "Total steps", network.decayStep, "time", round(gameTime, 2), "s", "\n", "0", round(moves[0] / totalMoves * 100.0, 2), "| 1", round(moves[1] / totalMoves * 100.0, 2), "| 2", round(moves[2] / totalMoves * 100.0, 2), "| 3", round(moves[3] / totalMoves * 100.0, 2), "epsilon", round(epsilon, 2))
            if won or network.nGames % 100 == 0:
                network.model.save()
            game.gameTime = time()


def getMove():
    move = [0,0,0,0]
    # [0 = UP,0 = DOWN,0 = LEFT,0 = RIGHT]
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        move[0] = 1
    elif keys[pygame.K_DOWN]:
        move[1] = 1
    elif keys[pygame.K_LEFT]:
        move[2] = 1
    elif keys[pygame.K_RIGHT]:
        move[3] = 1
    
    return move

train()
