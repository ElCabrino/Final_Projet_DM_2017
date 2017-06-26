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
	testingStart = timer.time()
	
	# we initialise any necessary variables
	successRate = 0
	confusionMatrix = np.zeros((5,5))

	# we go through the instances of testY
	for i in range(testY.shape[0]):
		# we get the actual score and guess the score
		actualScore = du.get_score(i, testY)
		guessedScore = guess_score()
		
		# if the guess is correct we increment the success rate
		if actualScore == guessedScore:
			successRate += 1

		# we increment the appropriate element of the confusion matrix 
		confusionMatrix[actualScore-1, guessedScore-1] += 1
		
	# we change our measures to percentiles	
	successRate /= testY.shape[0]
	
	# we stop the timer
	testingEnd = timer.time()

	# we find the total time for testing	
	testingTime = testingEnd-testingStart	
	
	return [successRate, confusionMatrix, testingTime]

[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

[sR,cM,teT] = get_performances(Y_test)

print(sR)
print(cM)
print(teT)
