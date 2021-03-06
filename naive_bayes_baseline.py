import numpy as np
import time as timer

import data_utils as du

def train(trainX, trainY):
	"""
	@param trainX : a matrix containing a bag of words representation of a text ie for a row each column represents the number of occurrences of a certain word
	@param trainY : a matrix of five columns for which every row has one column at 1 and the others at 0 indicating what the score is

	returns : the probabilities of a word being in a appearing in a certain class as well as class probability
	"""
	# we declare the necessary variables
	wordConditionalProb = np.zeros((5, trainX.shape[1]))
	scoreWordCount = np.zeros(5)
	scoreProb = np.zeros(5)

	# we go through the instances 
	for i in range(trainX.shape[0]):
		# we get the score
		score = du.get_score(i, trainY)

		# we modify the appropriate variables	
		wordConditionalProb[score-1, :] += trainX[i, :]
		scoreWordCount[score-1] += np.sum(trainX[i, :])
		scoreProb[score-1] += 1

	# we convert the counts to probabilites
	for i in range(5):
		wordConditionalProb[i, :] /= scoreWordCount[i]
		# if a count is at 0 we change it to a small value to not hinder classification
		for j in range(wordConditionalProb.shape[1]):
			if wordConditionalProb[i, j] == 0:
				wordConditionalProb[i,j] = 0.000001

	scoreProb /= trainX.shape[0]
	
	return [wordConditionalProb, scoreProb]

def guess_score(xRow, wordConditionalProb, scoreProb):
	"""
	@param xRow : the instance in bag of words format
	@param wordConditionalProb : a matrix containing the probabilities of a word appearing in a class
	@param scoreProb : score distribution

	returns : an estimated score
	"""
	# we start by taking the class probability
	scoreProbEstimate = np.copy(scoreProb)
	
	# we then multiply by the probability of the word being in the class if it is present (we the number of occurrences as an exponent)
	for i in range(len(xRow)):
		if xRow[i] != 0:
			for j in range(5):
				scoreProbEstimate[j] *= np.power(wordConditionalProb[j, i], xRow[i])
	
	# we find the score that best fits the instance
	score = 0
	maxScoreProb = 0

	for i in range(5):
		if scoreProbEstimate[i] > maxScoreProb:
			score = i+1
			maxScoreProb = scoreProbEstimate[i]
	
	return score

def get_performances(reviewTrainX, reviewTestX, titleTrainX, titleTestX, trainY, testY, reviewRatio):
	"""
	@param reviewTrainX : a matrix containing the training instances' review content in bag of words format
	@param reviewTestX : a matrix similar to reviewTrainX but with different instances
	@param titleTrainX : a matrix containing the training instances' title content in bag of words format
	@param titleTestX : a matrix containing the testing instances' title content in bag of words format
	@param trainY : a matrix containing the score of trainX's instances
	@param testY : a matrix containing the scores of the testX instances
	@param reviewRatio : a number between 0 and 1 indicating the importance given to review content when guessing a score

	returns : the model's raw success rate, its confusion matrix and the training and testing times
	"""
	# we do a timed training
	trainingStart = timer.time()

	# we train aspects if they will be used to guess
	if reviewRatio != 0:
		[reviewWordConditionalProb, reviewScoreProb] = train(reviewTrainX, trainY)
	if reviewRatio != 1:
		[titleWordConditionalProb, titleScoreProb] = train(titleTrainX, trainY)

	trainingEnd = timer.time()

	trainingTime = trainingEnd-trainingStart
	
	# we do a timed testing to find the performance measures
	testingStart = timer.time()

	successRate = 0
	confusionMatrix = np.zeros((5,5))

	for i in range(reviewTestX.shape[0]):
		actualScore = du.get_score(i, testY)
		# we compute the guesses if they are to be taken into account
		reviewGuessedScore = 0
		if reviewRatio != 0:
			reviewGuessedScore = guess_score(reviewTestX[i, :], reviewWordConditionalProb, reviewScoreProb)
		titleGuessedScore = 0
		if reviewRatio != 1:
			titleGuessedScore = guess_score(titleTestX[i, :], titleWordConditionalProb, titleScoreProb)

		guessedScore = int(round(reviewRatio*reviewGuessedScore+(1-reviewRatio)*titleGuessedScore))
	
		if actualScore == guessedScore:
			successRate += 1

		confusionMatrix[actualScore-1, guessedScore-1] += 1	

	successRate /= reviewTestX.shape[0]
	
	testingEnd = timer.time()

	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

"""
[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

# le dernier param est un nombre qui varie entre 0 et 1
[sR,cM,trT,teT] = get_performances(Xreview_train,Xreview_test,Xtitle_train,Xtitle_test,Y_train,Y_test,0)

print(sR)
print(cM)
print(trT)
print(teT)
"""

