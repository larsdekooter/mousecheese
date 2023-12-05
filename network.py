import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
from game import Game
import math
import numpy as np
import random
import data

class LinearQnet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.layer1 = nn.Linear(inputSize, hiddenSize)
        self.layer2 = nn.Linear(hiddenSize, hiddenSize)
        self.layer3 = nn.Linear(hiddenSize, outputSize)
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.sigmoid(self.layer1(x))
        x = F.tanh(self.layer2(x))
        return F.sigmoid(self.layer3(x))

    def save(self, filename="model.pth"):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        
        filename = os.path.join(modelFolderPath, filename)
        torch.save(self.state_dict(), filename)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def trainStep(self, state, action, reward, nextState, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        nextState = torch.tensor(np.array(nextState), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Qnew = reward[idx]
            if not done[idx]:
                Qnew = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Qnew

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class Network:
    def __init__(self):
        self.nGames = 0
        self.gamma = data.gamma
        self.memory = deque(maxlen=100_000)
        self.model = LinearQnet(9, data.hiddenSize, 4)
        self.model.load_state_dict(torch.load("./model/model.pth")) if os.path.exists(
            "./model/model.pth"
        ) else None
        self.trainer = QTrainer(self.model, lr=data.lr, gamma=self.gamma)
        self.maxEpsilon = data.maxEpsilon
        self.minEpsilon = data.minEpsilon
        self.decayRate = data.decayRate
        self.decayStep = 0
        self.aiSteps = 0
        self.randomSteps = 0
        self.moves = [0,0,0,0]

    def getState(self, game: Game):
        distanceToCheese = game.getDistanceToCheese()
        aroundLocations = [
            (game.mouse.x+100, game.mouse.y), # right
            (game.mouse.x-100, game.mouse.y), # left
            (game.mouse.x, game.mouse.y - 100), # up
            (game.mouse.x, game.mouse.y + 100), # down
        ]

        state = [
            # Cheese direction
            game.mouse.x < 700,
            game.mouse.x > 700,
            game.mouse.y < 700,
            game.mouse.y > 700,

            # Danger around
            aroundLocations[0] in data.catPositions,
            aroundLocations[1] in data.catPositions,
            aroundLocations[2] in data.catPositions,
            aroundLocations[3] in data.catPositions,   

            distanceToCheese 
        ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def trainLongMemory(self):
        if len(self.memory) > data.batchSize:
            miniSample = random.sample(self.memory, data.batchSize)
        else:
            miniSample = self.memory
        
        states, actions, rewards, nextStates, dones = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, nextStates, dones)

    def trainShortMemory(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)

    def getAction(self, state):
        if self.nGames < data.testLength:
            self.decayStep+=1
            epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(
                -self.decayRate * self.decayStep
            )
            final_move = [0,0,0,0]
            if np.random.rand() < epsilon:
                move = random.randint(0,3)
                final_move[move] = 1
                self.randomSteps +=1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                self.aiSteps += 1
            return final_move
        else:
            self.model.eval()
            with torch.no_grad():
                qValues = self.model(state)
            
            qValuesnp = qValues.numpy()
            #print("QValues: ", qValuesnp)
            action = torch.argmax(qValues).item()
            final_move = [0,0,0,0]
            final_move[action] = 1
            self.moves[action] += 1
            self.aiSteps +=1
            return final_move
