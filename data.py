import math
catPositions = [(100, 100), (200, 100), (300, 100), (500, 100), (600, 100), (700, 100), (0, 200), (500, 200), (100, 300), (300, 300), (400, 300), (500, 300), (400, 400), (400, 500), (400, 600), (600, 600), (600, 700)]
cooldown = 0.1
def getDistanceReward(distance):
    return 1.1**-distance
cheeseReward = 1000
rewardNerf = 5
gamma = 0.9
hiddenSize = 256
lr = 0.0001
maxEpsilon = 1
minEpsilon = 0.01
decayRate = 0.01
batchSize = 64
random = 200