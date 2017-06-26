from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import time as timer

import data_utils as du

def create_average_matrix(matrix, Z):
	averageMatrix = np.zeros([matrix.shape[0], Z.shape[1]])
	lineIt=0
	for xRow in matrix:
		# we declare necessary variables
		averageVector = np.zeros(Z.shape[1])
		vectorCount = 0
		
		# we sum up the vectors of an instance
		for i in range(len(xRow)):
			if xRow[i] != 0:
				averageVector += xRow[i]*Z[i, :]
				vectorCount += xRow[i]

		# we get the average vector for the instance
		if np.count_nonzero(averageVector) != 0:
			averageVector /= vectorCount

		averageMatrix[lineIt, :] = averageVector
		lineIt+=1
	return averageMatrix


def get_performances(reviewTrainX, reviewTestX, titleTrainX, titleTestX, trainY, testY, Z, reviewRatio):

	#converting Y matrices to vectors
	trainY = du.Y_to_vect(trainY)
	testY = du.Y_to_vect(testY)

	#creating the average matrices (aM) considering the reviews and the dictionnary Z:

	aM_review_train = create_average_matrix(reviewTrainX, Z)
	aM_title_train = create_average_matrix(titleTrainX, Z)
	aM_review_test = create_average_matrix(reviewTestX, Z)
	aM_title_test = create_average_matrix(titleTestX, Z)

	#we apply one linear regression per class vs the rest
	regr_rev = linear_model.LinearRegression()
	regr_title = linear_model.LinearRegression()

	trainingStart = timer.time()

	#training the 2 models
	if reviewRatio!= 0:
		fit_rev = OneVsRestClassifier(regr_rev).fit(aM_review_train, trainY)
	if reviewRatio!=1:
		fit_title = OneVsRestClassifier(regr_title).fit(aM_title_train, trainY)

	trainingEnd = timer.time()

	testingStart = timer.time()

	#testing the model
	if reviewRatio!=0:
		Y_pred_rev = fit_rev.predict(aM_review_test)
	if reviewRatio!=1:
		Y_pred_title = fit_title.predict(aM_title_test)


	#computing the confusion matrix and the score:
	#buidling the confusion matrix
	successRate = 0
	confusionMatrix = np.zeros((5,5))

	for i in range(reviewTestX.shape[0]):
		actualScore = int(testY[i])
		reviewGuessedScore = 0
		if reviewRatio!=0:
			reviewGuessedScore = int(Y_pred_rev[i])
		titleGuessedScore = 0
		if reviewRatio!=1:
			titleGuessedScore = int(Y_pred_title[i])

		guessedScore = int(round(reviewRatio*reviewGuessedScore+(1-reviewRatio)*titleGuessedScore))

		if actualScore == guessedScore:
			successRate += 1

		# we increment the adequate element of the confusion matrix
		confusionMatrix[actualScore-1, guessedScore-1] += 1

	successRate /= reviewTestX.shape[0]

	testingEnd = timer.time()

	trainingTime = trainingEnd-trainingStart

	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

"""
[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

[score_rev, cM_rev, trT_rev, teT_rev] = get_performances(Xtitle_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test, Z, 0)

print('Results:')
print(score_rev)
print(cM_rev)
print(trT_rev)
print(teT_rev)
"""




