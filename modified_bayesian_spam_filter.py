import numpy as np
import time as timer
import warnings

import data_utils as du

def train(trainX, trainY):
	"""
	@param trainX : a matrix in bag of words format
	@param trainY : a matrix containing the scores

	returns : class posterior, score probability as well as word counts
	"""
	# we declare necessary variables
	wordConditionalProbs = np.zeros((5, trainX.shape[1]))
	scoreProbs = np.zeros(5)
	wordCounts = np.zeros(trainX.shape[1])

	# we get conditional probabilities, these are the probability of a word being in a certain type of message
	for i in range(trainX.shape[0]):
		score = du.get_score(i, trainY)
			
		for j in range(trainX.shape[1]):
			if trainX[i, j] != 0:
				wordConditionalProbs[score-1, i] += 1
		
		# we also find score probabilities
		scoreProbs[score-1] += 1
		
		# we also find word count
		wordCounts += trainX[i, :]	

	# we convert elements to probabilites
	for i in range(5):
		wordConditionalProbs[i, :] /= scoreProbs[i]

	scoreProbs /= trainX.shape[0]

	# if we have 0 prob we put a small value
	for i in range(wordConditionalProbs.shape[0]):
		for j in range(wordConditionalProbs.shape[1]):
			if wordConditionalProbs[i, j] == 0:
				wordConditionalProbs[i, j] = 1e-4

	return [wordConditionalProbs, scoreProbs, wordCounts]

def get_class_prior_biased(classPosteriors, classProbs, desiredClassIndex):
	"""
	@param classPosteriors : word distribution given class
	@param classProbs : class distribution
	@param desiredClassIndex : indicates the class we want as prior

	returns : class prior with bias (takes into account class distribution)
	"""
	# get numerator
	numerator = classPosteriors[desiredClassIndex]*classProbs[desiredClassIndex]
	
	# get denominator
	denominator = 0
	for i in range(5):
		denominator += classPosteriors[i]*classProbs[i]

	# find class prior
	classPrior = numerator/denominator
	
	return classPrior

def get_class_prior_unbiased(classPosteriors, desiredClassIndex):
	"""
	@param classPosteriors : word distribution given class
	@param desiredClassIndex : indicates the desired class

	returns : class prior without bias (assumes that words are evenly distributed)
	"""
	# we get the numerator
	numerator = classPosteriors[desiredClassIndex]
	
	# we get the denominator
	denominator = np.sum(classPosteriors)

	# we get the class prior
	classPrior = numerator/denominator

	return classPrior

def get_corrected_class_prior(classStrength, classProbs, wordCount, classPosteriors, desiredClassIndex, biasPred):
	"""
	@param classStrength : indicates importance attached to actual class distribution
	@param classProbs : class distribution
	@param wordCount : word count
	@param classPosteriors : word distribution given class
	@param desiredClassIndex : indicates the desired class
	@param biasedPred : indicates whether we take class distribution into account when computing the initial bias
	
	returns : the corrected class prior for rare words
	"""
	# we get the adapted class prior
	if biasPred:
		classPrior = get_class_prior_biased(classPosteriors, classProbs, desiredClassIndex)
	else:
		classPrior = get_class_prior_unbiased(classPosteriors, desiredClassIndex)

	# we get the numerator
	numerator = classStrength*classProbs[desiredClassIndex]+wordCount*classPrior
	
	# we get the denominator
	denominator = classStrength+wordCount

	# we get the new prior if there is no information the class prior is the original class prior
	if denominator != 0:
		correctedClassPrior = numerator/denominator
	else:
		correctedClassPrior = classPrior

	return correctedClassPrior

def get_row_class_probability(xRow, classStrength, classProbs, wordCounts, classPosteriors, desiredClassIndex, biasPred):
	"""
	@param xRow : the bag of words form of an instance
	@param classStrength : the strength of a class distribution
	@param classProbs : the class distribution
	@param wordCounts : the word counts of the learning process
	@param classPosteriors : word distribution given a class
	@param desiredClassIndex : indicates the desired class
	@param biasPred : indicates whether we use biased initial prior class

	returns : the probability of an instance being of a certain class
	"""
	# we declare necessary variables
	eta = 0

	# we go through the words
	for i in range(len(xRow)):
		# if a word is present
		if xRow[i] != 0:
			# we get the class prior
			classPrior = get_corrected_class_prior(classStrength, classProbs, wordCounts[i], classPosteriors[:, i], desiredClassIndex, biasPred)
			# we add the the word probability to the class
			eta += np.log(1-classPrior)-np.log(classPrior)
	
	# we get the probability checking for overflow
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		try:
			classProb = 1/(1+np.exp(eta))
		except Warning as e:
			classProb = 0

	return classProb

def guess_score(xRow, classStrength, classProbs, wordCounts, classPosteriors, biasPred):
	"""
	@param xRow : the bag of words form of an instance
	@param classStrength : the strength of a class distribution
	@param classProbs : the class distribution
	@param wordCounts : the word counts of the learning process
	@param classPosteriors : word distribution given a class
	@param biasPred : indicates whether we use biased initial prior class

	returns : the estimated score
	"""
	# declare necessary variables
	rowClassProbs = np.zeros(5)
	score = 0
	maxProb = 0	
	
	# we get all class probabilites and find the maximal probability
	for i in range(5):
		rowClassProbs[i] = get_row_class_probability(xRow, classStrength, classProbs, wordCounts, classPosteriors, i, biasPred)
		if rowClassProbs[i] > maxProb:
			score = i+1
			maxProb = rowClassProbs[i]
	
	return score

def get_performances(reviewTrainX, reviewTestX, titleTrainX, titleTestX, trainY, testY, reviewRatio, classStrength, biasPred):
	"""
	@param reviewTrainX : a matrix containing the training instances' review content in bag of words format
	@param reviewTestX : a matrix similar to reviewTrainX but with different instances
	@param titleTrainX : a matrix containing the training instances' title content in bag of words format
	@param titleTestX : a matrix containing the testing instances' title content in bag of words format
	@param trainY : a matrix containing the score of trainX's instances
	@param testY : a matrix containing the scores of the testX instances
	@param reviewRatio : a number between 0 and 1 indicating the importance given to review content when guessing a score
	@param biasPred : indicates whether initial prior probabilites should be influenced by class distribution
	
	returns : the model's raw success rate, its confusion matrix and the training and testing times
	"""
	# we do a timed training
	trainingStart = timer.time()

	# we train aspects if they will be used to guess
	if reviewRatio != 0:
		[reviewWordConditionalProbs, reviewScoreProbs, reviewWordCounts] = train(reviewTrainX, trainY)
	if reviewRatio != 1:
		[titleWordConditionalProbs, titleScoreProbs, titleWordCounts] = train(titleTrainX, trainY)

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
			reviewGuessedScore = guess_score(reviewTestX[i, :], classStrength, reviewScoreProbs, reviewWordCounts, reviewWordConditionalProbs, biasPred)
		titleGuessedScore = 0
		if reviewRatio != 1:
			titleGuessedScore = guess_score(titleTestX[i, :], classStrength, titleScoreProbs, titleWordCounts, titleWordConditionalProbs)

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

# les derniers params sont un nombre qui varie entre 0 et 1, est un nombre qui peut avoir n'importe quelle valeur positive est un booleen
[sR,cM,trT,teT] = get_performances(Xreview_train,Xreview_test,Xtitle_train,Xtitle_test,Y_train,Y_test,1,10,True)

print(sR)
print(cM)
print(trT)
print(teT)
"""
