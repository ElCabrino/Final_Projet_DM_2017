from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import time as timer

import data_utils as du

def get_performances(trainX, trainY, testX, testY, Z):

	#passing Y matrices to vectors
	trainY = du.Y_to_vect(trainY)
	testY = du.Y_to_vect(testY)

	averageMatrix = np.zeros([trainX.shape[0], Z.shape[1]])
	lineIt=0
	for xRow in trainX:
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

	averageMatrixTest = np.zeros([testX.shape[0], Z.shape[1]])
	lineIt=0
	for xRow in testX:
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

		averageMatrixTest[lineIt, :] = averageVector
		lineIt+=1

	#we apply one linear regression per class vs the rest
	regr = linear_model.LinearRegression()

	trainingStart = timer.time()

	#training the model
	fit = OneVsRestClassifier(regr).fit(averageMatrix, trainY)

	trainingEnd = timer.time()

	testingStart = timer.time()

	#testing the model
	score = fit.score(averageMatrixTest, testY)

	testingEnd = timer.time()

	#computing the confusion matrix:
	#getting the prediction
	Y_pred = fit.predict(averageMatrixTest)

	#buidling the confusion matrix
	confusionMatrix = np.zeros((5,5))

	for i in range(len(Y_pred)):
		actualScore = int(testY[i])
		guessedScore = int(Y_pred[i])

		# we increment the adequate element of the confusion matrix
		confusionMatrix[actualScore-1, guessedScore-1] += 1

	#converting to percentile
	confusionMatrix /= testX.shape[0]

	trainingTime = trainingEnd-trainingStart

	testingTime = testingEnd-testingStart

	return [score, confusionMatrix, trainingTime, testingTime]


[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

[score_rev, cM_rev, trT_rev, teT_rev] = get_performances(Xreview_train, Y_train, Xreview_test, Y_test, Z)
[score_rev_title, cM_rev_title, trT_rev_title, teT_rev_title] = get_performances(Xreview_train+Xtitle_train, Y_train, Xreview_test+Xtitle_test, Y_test, Z)
[score_title, cM_title, trT_title, teT_title] = get_performances(Xtitle_train, Y_train, Xtitle_test, Y_test, Z)

print('Results with reviews only:')
print(score_rev)
print(cM_rev)
print(trT_rev)
print(teT_rev)
print('Results with reviews and titles:')
print(score_rev_title)
print(cM_rev_title)
print(trT_rev_title)
print(teT_rev_title)
print('Results with titles only:')
print(score_title)
print(cM_title)
print(trT_title)
print(teT_title)



