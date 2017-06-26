import numpy as np
from scipy import spatial
import time as timer

import data_utils as du

def train(trainX, trainY, Z):
	"""
	@param trainX : a matrix containing a bag of words representation of a text ie for a row each column  represents the number of occurrences of a certain word
	@param trainY : a matrix of five columns five columns for which every row has one column at 1 and the others at 0 indicating what the score is
	@param Z : a matrix : a matrix which every row is a vector representing a word

	returns the normalised sum of vectors for every score as well as the covariance scores
	"""
	# we declare the necessary variable
	scoreVectors = np.zeros((5,Z.shape[1]))
	instanceCounts = np.zeros(5)

	# we go through the instances
	for i in range(trainX.shape[0]):
		# we get the instance's score
		score = du.get_score(i, trainY)
		instanceVector = np.zeros(Z.shape[1])
		for j in range(trainX.shape[1]):
			# we sum up the present word vectors
			if trainX[i, j] != 0:
				instanceVector += trainX[i, j]*Z[j, :]
		if np.linalg.norm(instanceVector) != 0:
			scoreVectors[score-1, :] += instanceVector/np.linalg.norm(instanceVector)
		instanceCounts[score-1] += 1
	
	# we normalise the vectors
	for i in range(5):
		scoreVectors[i, :] /= np.linalg.norm(scoreVectors[i, :])

	# we get covariances for mahalanobis
	instanceMatrices = []
	for i in range(5):
		instanceMatrices.append(np.zeros((int(instanceCounts[i]), Z.shape[1])))

	instanceIndexes = np.zeros(5)

	for i in range(trainX.shape[0]):
		score = du.get_score(i, trainY)
		# we then go through the instance to get the scores' sum of average vectors
		instanceVector = np.zeros(Z.shape[1])
		for j in range(trainX.shape[1]):
			if trainX[i, j] != 0:
				instanceVector += trainX[i, j]*Z[j, :]
		# we assign the element to the right part of the right matrix
		if np.linalg.norm(instanceVector) != 0:
			instanceMatrices[score-1][int(instanceIndexes[score-1]), :] = instanceVector/np.linalg.norm(instanceVector)
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
	@param scoreVectors : a matrix for which every row is the normalised sum of vectors present in a score
	@param trainInstanceCovs : the covariance matrices for training instances of different scores

	returns : an estimate of the score based on the nearest normalised score vector
	"""
	# we declare the necessary variable
	normalisedVector = np.zeros(Z.shape[1])
	
	# we sum up the vectors of the instance
	for i in range(len(xRow)):
		if xRow[i] != 0:
			normalisedVector += xRow[i]*Z[i, :]

	# we normalise the instance
	if np.linalg.norm(normalisedVector) != 0:
		normalisedVector /= np.linalg.norm(normalisedVector)

	# we find the nearest normalised score vector to the normalised instance vector
	score = 0
	minScoreDist = np.inf
	
	for i in range(5):
		if spatial.distance.mahalanobis(normalisedVector, scoreVectors[i, :], np.linalg.inv(trainInstanceCovs[i])) < minScoreDist:
			score = i+1
			minScoreDist = spatial.distance.mahalanobis(normalisedVector, scoreVectors[i, :], np.linalg.inv(trainInstanceCovs[i]))
	
	return score

def get_performances(reviewTrainX, reviewTestX, titleTrainX, titleTestX, trainY, testY, Z, reviewRatio):
	"""
	@param reviewTrainX : a matrix containing the training instances' review content in bag of words format
	@param reviewTestX : a matrix the test instances' review content in bag of words format, the instances are different from those in trainX
	@param titleTrainX : a matrix containing training instances' title content in bag of words format
	@param titleTestX : a matrix containing testing instances' title content in bag of words format
	@param trainY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is
	@param testY _ a matrix similar to trainY containing different instances
	@param Z : a matrix containing a vector representation of the words in adictionnary
	@param reviewRatio : a number between 0 and 1 indicating the importance attached to review content when guessing
	
	returns : the model's raw success rate, its confusion matrix and the training and testing times
	"""
	# we start the training timer
	trainingStart = timer.time()
	
	# we get the normalised score vectors and covariance matrices
	if reviewRatio != 0:
		[reviewScoreVectors, reviewTrainInstanceCovs] = train(reviewTrainX, trainY, Z)
	if reviewRatio != 1:
		[titleScoreVectors, titleTrainInstanceCovs] = train(titleTrainX, trainY, Z)

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
	for i in range(reviewTestX.shape[0]):
		# we get the actual score and the estimated score
		actualScore = du.get_score(i, testY)
		reviewGuessedScore = 0
		if reviewRatio != 0:
			reviewGuessedScore = guess_score(reviewTestX[i, :], Z, reviewScoreVectors, reviewTrainInstanceCovs)
		titleGuessedScore = 0
		if reviewRatio != 1:
			titleGuessedScore = guess_score(titleTestX[i, :], Z, titleScoreVectors, titleTrainInstanceCovs)

		# the guessed score is a combination of the title and the review
		guessedScore = int(round(reviewRatio*reviewGuessedScore+(1-reviewRatio)*titleGuessedScore))

		# if the guessed score is correct we increment the the success rate
		if actualScore == guessedScore:
			successRate += 1

		# we increment the adequate element of the confusion matrix
		confusionMatrix[actualScore-1, guessedScore-1] += 1

	# we convert the variables to percentiles
	successRate /= reviewTestX.shape[0]
	confusionMatrix /= reviewTestX.shape[0]

	# we stop the testing timer
	testingEnd = timer.time()

	# we get the testing time
	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

[sR,cM,trT,teT] = get_performances(Xreview_train,Xreview_test, Xtitle_train, Xtitle_test,Y_train,Y_test,Z,0)

print(sR)
print(cM)
print(trT)
print(teT)
