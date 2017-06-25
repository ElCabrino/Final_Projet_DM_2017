import numpy as np
import time as timer

import data_utils as du

def train(trainX, trainY, Z):
	"""
	@param trainX : a matrix containing a bag of words representation of a text ie for a row each column  represents the number of occurrences of a certain word
	@param trainY : a matrix of five columns five columns for which every row has one column at 1 and the others at 0 indicating what the score is
	@param Z : a matrix : a matrix which every row is a vector representing a word

	returns the normalised sum of vectors for every score
	"""
	# we declare the necessary variable
	scoreVectors = np.zeros((5,Z.shape[1]))

	# we go through the instances
	for i in range(trainX.shape[0]):
		# we get the instance's score
		score = du.get_score(i, trainY)
		instanceVector = np.zeros(Z.shape[1])
		for j in range(trainX.shape[1]):
			# we sum up the present word vectors
			if trainX[i, j] != 0:
				instanceVector += trainX[i, j]*Z[j, :]
		scoreVectors[score-1, :] += instanceVector/np.linalg.norm(instanceVector)
	
	# we normalise the vectors
	for i in range(5):
		scoreVectors[i, :] /= np.linalg.norm(scoreVectors[i, :])

	return scoreVectors

def guess_score(xRow, Z, scoreVectors):
	"""
	@param xRow : a bag of words representation of an instance
	@param Z : a matrix for which every row is a vector representing a word
	@param scoreVectors : a matrix for which every row is the normalised sum of vectors present in a score

	returns : an estimate of the score based on the nearest normalised score vector
	"""
	# we declare the necessary variable
	averageVector = np.zeros(Z.shape[1])
	
	# we sum up the vectors of the instance
	for i in range(len(xRow)):
		if xRow[i] != 0:
			averageVector += xRow[i]*Z[i, :]

	# we normalise the instance
	averageVector /= np.linalg.norm(averageVector)

	# we find the nearest normalised score vector to the normalised instance vector
	score = 1
	minScoreDist = np.linalg.norm(averageVector-scoreVectors[0, :])
	
	for i in range(5):
		if np.linalg.norm(averageVector-scoreVectors[i, :]) < minScoreDist:
			score = i+1
			minScoreDist = np.linalg.norm(averageVector-scoreVectors[i, :])
	
	return score

def get_performances(trainX, testX, trainY, testY, Z):
	"""
	@param trainX : a matrix containing the training instances in bag of words format
	@param testX : a matrix the test instances in bag of words format, the instances are different from those in trainX
	@param trainY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is
	@param testY _ a matrix similar to trainY containing different instances
	@param Z : a matrix containing a vector representation of the words in adictionnary

	returns : the model's raw success rate, its confusion matrix and the training and testing times
	"""
	# we start the training timer
	trainingStart = timer.time()
	
	# we get the normalised score vectors
	scoreVectors = train(trainX, trainY, Z)

	# we stop the training timer
	trainingEnd = timer.time()
	
	# we get the training time
	trainingTime = trainingEnd-trainingStart

	# we start the testing timer
	testingStart = timer.time()

	# we declare the necessary variables for performace evaluation
	successRate = 0
	confusionMatrix = np.zeros((5,5))

	# we go through the test instances
	for i in range(testX.shape[0]):
		# we get the actual score and the estimated score
		actualScore = du.get_score(i, testY)
		guessedScore = guess_score(testX[i, :], Z, scoreVectors)

		# if the guessed score is correct we increment the the success rate
		if actualScore == guessedScore:
			successRate += 1

		# we increment the adequate element of the confusion matrix
		confusionMatrix[actualScore-1, guessedScore-1] += 1

	# we convert the variables to percentiles
	successRate /= testX.shape[0]
	confusionMatrix /= testX.shape[0]

	# we stop the testing timer
	testingEnd = timer.time()

	# we get the testing time
	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

[sR,cM,trT,teT] = get_performances(Xreview_train,Xreview_test,Y_train,Y_test,Z)

print(sR)
print(cM)
print(trT)
print(teT)
