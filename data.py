catPositions = [(100, 100), (200, 100), (300, 100), (500, 100), (600, 100), (700, 100), (0, 200), (500, 200), (100, 300), (300, 300), (400, 300), (500, 300), (400, 400), (400, 500), (400, 600), (600, 600), (600, 700)]
cooldown = 0
cheeseReward = 1000
rewardNerf = 5
gamma = 0.9
hiddenSize = 64  # Increased hidden layer size for more capacity
lr = 0.0001  # Adjusted learning rate for stability
maxEpsilon = 1
minEpsilon = 0.01
decayRate = 0.0001
batchSize = 64  # Adjusted batch size for efficiency
random = 200

def getDistanceReward(distance):
    return 1.02 ** -distance  # Slightly increased reward decay

def getEfficiencyPenalty(distance):
    return 0.005 * distance  # Reduced efficiency penalty for exploration

