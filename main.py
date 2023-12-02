from game import Game
import pygame
from time import sleep
from network import Network

def train():
    network = Network()
    game = Game()
    while True:
        state = network.getState(game)
        move = network.getAction(state)

        done, reward, won = game.step(move)
        stateNew = network.getState(game)

        network.trainShortMemory(state, move, reward, stateNew, done)
        network.remember(state, move, reward, stateNew, done)


        if done:
            game.reset()
            network.nGames+=1
            network.trainLongMemory()

            print("game", network.nGames, "won", won)
            if won:
                network.model.save()


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