import numpy as np
import time as timer

import data_utils as du

def train(trainY):
	"""
	@param trainY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is

	returns : the score that is most prevalent among the instances in trainY
	"""
	# we initilise the counters for each score
	scoreCounter = [0, 0, 0, 0, 0]
	
	# we count the number of occurences of each score
	for i in range(trainY.shape[0]):
		scoreCounter[du.get_score(i, trainY)-1] += 1

	# we initialise variables necessary to finding the majority score
	majorityScore = 0
	majorityScoreCount = 0	

	# we find the most prevalent score
	for i in range(5):
		if scoreCounter[i] > majorityScoreCount:
			majorityScore = i+1
			majorityScoreCount = scoreCounter[i]

	return majorityScore

def get_performances(trainY, testY):
	"""
	@param trainY : a score matrix
	@param testY : a score matrix containing instances not present in trainY
	
	returns : the model's raw success rate, its confusion matrix as well the training and test times
	"""
	# we start the training timer
	trainingStart = timer.time()

	# we train the model
	majorityScore = train(trainY)

	# we end the training timer
	trainingEnd = timer.time()

	# we find the total time for training
	trainingTime = trainingEnd-trainingStart

	# we start the testing timer
	testingStart = timer.time()
	
	# we initilise any necessary variables
	successRate = 0
	confusionMatrix = np.zeros((5,5))

	# we go through the instances of testY
	for i in range(testY.shape[0]):
		actualScore = du.get_score(i, testY)
		
		# if the score is the majority score we increment the success rate
		if majorityScore == actualScore:
			successRate += 1

		# we increment the appropriate element of the confusion matrix
		confusionMatrix[actualScore-1, majorityScore-1] += 1
	
	# we convert the measures to percentiles
	successRate /= testY.shape[0]
	confusionMatrix /= testY.shape[0]  

	# we stop the testing timer
	testingEnd = timer.time()

	# we find the total testing time
	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

Y = du.get_Y('working_dir/ratings.txt')

[sR,cM,trT,teT] = get_performances(Y,Y)

print(sR)
print(cM)
print(trT)
print(teT)
