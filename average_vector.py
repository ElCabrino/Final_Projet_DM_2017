import numpy as np
from scipy import spatial
import time as timer

import data_utils as du

def train(trainX, trainY, Z):
	"""
	@param trainX : a matrix containing a bag of words representation of a text ie for a row each column represents the number of occurrences of a certain word
	@param trainY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is
	@param Z : a matrix for which every row is a vector representing a word

	returns : the average vector for every score as well as the covariance matrices for each score
	"""
	# we start by declaring the necessary variables
	scoreVectors = np.zeros((5, Z.shape[1]))
	instanceCounts = np.zeros(5)

	# we then go through the instances
	for i in range(trainX.shape[0]):
		# we get the instance's score
		score = du.get_score(i, trainY)
		# we then go through the instance to get the scores' sum of average vectors
		instanceVector = np.zeros(Z.shape[1])
		vectorCount = 0
		for j in range(trainX.shape[1]):
			if trainX[i, j] != 0:
				instanceVector += trainX[i, j]*Z[j, :]
				vectorCount += trainX[i, j]

		if vectorCount != 0:
			scoreVectors[score-1, :] += instanceVector/vectorCount
		instanceCounts[score-1] += 1
	
	# we get the average vector of a score
	for i in range(5):
		scoreVectors[i, :] /= instanceCounts[i]

	# we get covariances for mahalanobis
	instanceMatrices = []
	for i in range(5):
		instanceMatrices.append(np.zeros((int(instanceCounts[i]), Z.shape[1])))

	instanceIndexes = np.zeros(5)

	for i in range(trainX.shape[0]):
		score = du.get_score(i, trainY)
		# we then go through the instance to get the scores' sum of average vectors
		instanceVector = np.zeros(Z.shape[1])
		vectorCount = 0
		for j in range(trainX.shape[1]):
			if trainX[i, j] != 0:
				instanceVector += trainX[i, j]*Z[j, :]
				vectorCount += trainX[i, j]
		# we assign the element to the right part of the right matrix
		if vectorCount != 0:
			instanceMatrices[score-1][int(instanceIndexes[score-1]), :] = instanceVector/vectorCount
		instanceIndexes[score-1] += 1
	
	# we get all the covariances
	instanceCovs = []
	
	for i in range(5):
		instanceCovs.append(np.cov(instanceMatrices[i], rowvar = 0))
	
	return [scoreVectors, instanceCovs]

def guess_score(xRow, Z, scoreVectors, trainInstanceCovs):
	"""
	@param xRow : a bag of words representation of an instance
	@param Z : a matrix for which every row is a vector representing a word
	@param scoreVectors : a matrix for which every row is the average vector for a score
	@param trainInstanceCovs : the covariance matrices for training instances of different scores
	
	returns : an estimate of the score based on the nearest average score vector to the average instance vector
	"""
	# we declare necessary variables
	averageVector = np.zeros(Z.shape[1])
	vectorCount = 0
	
	# we sum up the vectors of an instance
	for i in range(len(xRow)):
		if xRow[i] != 0:
			averageVector += xRow[i]*Z[i, :]
			vectorCount += xRow[i]

	# we get the average vector for the instance
	if vectorCount != 0:
		averageVector /= vectorCount

	# we find the nearest score average vector to the instance average vector
	score = 0
	minScoreDist = np.inf
	
	for i in range(5):
		if spatial.distance.mahalanobis(averageVector, scoreVectors[i, :], np.linalg.inv(trainInstanceCovs[i])) < minScoreDist:
			score = i+1
			minScoreDist = spatial.distance.mahalanobis(averageVector, scoreVectors[i, :], np.linalg.inv(trainInstanceCovs[i]))
	
	return score

def get_performances(reviewTrainX, reviewTestX, titleTrainX, titleTestX, trainY, testY, Z, reviewRatio):
	"""
	@param reviewTrainX : a matrix containing the training instances' review content in bag of words format
	@param reviewTestX : a matrix containing the test instances' review content in bag of words format, the instances are different from those of trainX
	@param titleTrainX : a matrix similar to reviewTrainX, but containing title content
	@param titleTestX : a matrix similar to titleTrainX, but containing different instances
	@param trainY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is
	@param testY : a matrix similar to trainY containing different instances	@param Z : a matrix containing a vector representation of the words in a dictionnary
	@param reviewRatio : a number between 0 and 1 indicating the importance we give to the review aspect of an instance
	
	returns : the model's raw success rate, its confusion matrix and the training and testing times
	"""
	# we start the training timer
	trainingStart = timer.time()
	
	# we get the average score vector and covariance if we take these elements into account
	if reviewRatio != 0:
		[reviewScoreVectors, reviewInstanceTrainCovs] = train(reviewTrainX, trainY, Z)
	if reviewRatio != 1:
		[titleScoreVectors, titleInstanceTrainCovs] = train(titleTrainX, trainY, Z)

	# we stop the training timer
	trainingEnd = timer.time()
	
	# we get the training time
	trainingTime = trainingEnd-trainingStart

	# we start the training time
	testingStart = timer.time()

	# we declare necessary variables for evaluation
	successRate = 0
	confusionMatrix = np.zeros((5,5))

	# we go through every instance
	for i in range(reviewTestX.shape[0]):
		# we get the actual score and the estimated score
		actualScore = du.get_score(i, testY)
		reviewGuessedScore = 0
		if reviewRatio != 0:
			reviewGuessedScore = guess_score(reviewTestX[i, :], Z, reviewScoreVectors, reviewInstanceTrainCovs)
		titleGuessedScore = 0
		if reviewRatio != 1:
			titleGuessedScore = guess_score(titleTestX[i, :], Z, titleScoreVectors, titleInstanceTrainCovs)

		# the guess is a mix between the title guess and the review guess
		guessedScore = int(round(reviewRatio*reviewGuessedScore+(1-reviewRatio)*titleGuessedScore))
		
		# if the score is correct we increment the success rate
		if actualScore == guessedScore:
			successRate += 1

		# we increment the adequate element of the confusion matrix
		confusionMatrix[actualScore-1, guessedScore-1] += 1

	# we convert the measures to percentiles
	successRate /= reviewTestX.shape[0]

	# we stop the testing timer
	testingEnd = timer.time()

	# we get the testing time
	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

"""
[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

# dernier param varie entre 0 et 1
[sR,cM,trT,teT] = get_performances(Xreview_train,Xreview_test,Xtitle_train,Xtitle_test,Y_train,Y_test,Z,0)

print(sR)
print(cM)
print(trT)
print(teT)
"""
