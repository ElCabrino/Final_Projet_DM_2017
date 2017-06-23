import random
import numpy as np
import time as timer

import data_utils as du

def guess_score():
	"""
	returns : a random integer between 1 and 5
	"""
	return random.randint(1,5)

def get_performances(testY):
	"""
	@param testY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is

	returns : the model's raw success rate, its confusion matrix and the time necessary to go through all the instances (rows) in testY
	"""
	# we start the timer
	start = timer.time()
	
	# we initialise any necessary variables
	successRate = 0
	confusionMatrix = np.zeros((5,5))

	# we go through the instances of testY
	for i in range(testY.shape[0]):
		# we guess the score and get the actual score
		guessedScore = guess_score()
		actualScore = du.get_score(i, testY)
		
		# if the guess is correct we increment the success rate
		if guessedScore == actualScore:
			successRate += 1

		# we increment the appropriate element of the confusion matrix 
		confusionMatrix[guessedScore-1, actualScore-1] += 1
		
	# we change our measures to percentiles	
	successRate /= testY.shape[0]
	confusionMatrix /= testY.shape[0]
	
	# we stop the timer
	end = timer.time()
	
	return [successRate, confusionMatrix, end-start]
