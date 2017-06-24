import random
import numpy as np
import time as timer

import data_utils as du

def train(trainY):
	"""
	@param trainY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is

	returns : a mapping enabling us to get a score from a number between 0 and 1 in a way that simulates score distribution
	"""
	# we initialise the score distribution measures
	scoreDistribution = np.zeros(5)
	
	# we count the number of occurrences for each score
	for i in range(trainY.shape[0]):
		scoreDistribution[du.get_score(i, trainY)-1] += 1

	# we convert the scores measures to percentiles
	scoreDistribution /= trainY.shape[0]

	# we initialise the mapping array
	scoreMapping = np.zeros(5)
	
	# we build the mapping array by adding the probability of the previous score's occurrence to the previous value in the mapping array
	for i in range(4):
		scoreMapping[i+1] = scoreMapping[i]+scoreDistribution[i]

	return scoreMapping

def guess_score(scoreMapping):
	"""
	@param scoreMapping : a mapping enabling us to get a score from a number between 0 and 1 in a way that simulates score distribution

	returns : an integer between 0 and 1
	"""
	# we get a random value between 0 and 1
	randVal = random.random()
	
	# we get the score corresponding to the random value
	score = 0	
	while scoreMapping[score] <= randVal:
		score += 1
		if score == 5:
			break
	
	return score

def get_performances(trainY, testY):
	"""
	@param trainY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is
	@param testY : a matrix similar to trainY only it contains different instances

	returns : the model's raw success rate, its confusion matrix as well as the training and test times
	"""
	# we start the training timer
	trainingStart = timer.time()
	
	# we train the model
	scoreMapping = train(trainY)
	
	# we stop the training timer
	trainingEnd = timer.time()

	# we get the training time
	trainingTime = trainingEnd-trainingStart

	# we start the testing timer
	testingStart = timer.time()

	# we initialise any necessary variables
	successRate = 0
	confusionMatrix = np.zeros((5,5))

	# we go through the instances in testY
	for i in range(testY.shape[0]):
		# we get the actual score and the model's guess
		actualScore = du.get_score(i, testY)
		guessedScore = guess_score(scoreMapping)
	
		# if the guess is correct we increment the success rate
		if actualScore == guessedScore:
			successRate += 1

		# we increment the appropriate value in the confusion matrix
		confusionMatrix[actualScore-1, guessedScore-1] += 1

	# we convert the measures into percentiles
	successRate /= testY.shape[0]
	confusionMatrix /= testY.shape[0]

	# we stop the testing timer
	testingEnd = timer.time()

	# we calculate the testing time
	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

[sR,cM,trT,teT] = get_performances(Y_test,Y_test)

print(sR)
print(cM)
print(trT)
print(teT)
