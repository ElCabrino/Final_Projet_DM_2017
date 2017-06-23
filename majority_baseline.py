import numpy as np
import time as timer

import data_utils as du

def train(trainY):
	scoreCounter = [0, 0, 0, 0, 0]
	
	for i in range(trainY.shape[0]):
		scoreCounter[du.get_score(i, trainY)-1] += 1

	majorityScore = 0
	majorityScoreCount = 0	

	for i in range(5):
		if scoreCounter[i] > majorityScoreCount:
			majorityScore = i+1
			majorityScoreCount = scoreCounter[i]

	return majorityScore

def get_performances(trainY, testY):
	trainingStart = timer.time()

	majorityScore = train(trainY)

	trainingEnd = timer.time()

	trainingTime = trainingEnd-trainingStart

	testingStart = timer.time()
	
	successRate = 0
	confusionMatrix = np.zeros((5,5))

	for i in range(testY.shape[0]):
		actualScore = du.get_score(i, testY)
		
		if majorityScore == actualScore:
			successRate += 1

		confusionMatrix[actualScore-1, majorityScore-1] += 1
	
	successRate /= testY.shape[0]
	confusionMatrix /= testY.shape[0]  

	testingEnd = timer.time()

	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

Y = du.get_Y('working_dir/ratings.txt')

[sR,cM,trT,teT] = get_performances(Y,Y)

print(sR)
print(cM)
print(trT)
print(teT)
