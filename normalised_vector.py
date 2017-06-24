import numpy as np
import time as timer

import data_utils as du

def train(trainX, trainY, Z):
	scoreVectors = np.zeros((5,Z.shape[1]))

	for i in range(trainX.shape[0]):
		score = du.get_score(i, trainY)
		for j in range(trainX.shape[1]):
			if trainX[i, j] != 0:
				scoreVectors[score-1, :] += trainX[i, j]*Z[j, :]
	
	for i in range(5):
		scoreVectors[i, :] /= np.linalg.norm(scoreVectors[i, :])

	return scoreVectors

def guess_score(xRow, Z, scoreVectors):
	averageVector = np.zeros(Z.shape[1])
	
	for i in range(len(xRow)):
		if xRow[i] != 0:
			averageVector += xRow[i]*Z[i, :]

	averageVector /= np.linalg.norm(averageVector)

	score = 1
	minScoreDist = np.linalg.norm(averageVector-scoreVectors[0, :])
	
	for i in range(4):
		if np.linalg.norm(averageVector-scoreVectors[i+1, :]) < minScoreDist:
			score = i+2
			minScoreDist = np.linalg.norm(averageVector-scoreVectors[i+1, :])
	
	return score

def get_performances(trainX, testX, trainY, testY, Z):
	trainingStart = timer.time()
	
	scoreVectors = train(trainX, trainY, Z)

	trainingEnd = timer.time()
	
	trainingTime = trainingEnd-trainingStart

	testingStart = timer.time()

	successRate = 0
	confusionMatrix = np.zeros((5,5))

	for i in range(testX.shape[0]):
		actualScore = du.get_score(i, testY)
		guessedScore = guess_score(testX[i, :], Z, scoreVectors)

		if actualScore == guessedScore:
			successRate += 1

		confusionMatrix[actualScore-1, guessedScore-1] += 1

	successRate /= testX.shape[0]
	confusionMatrix /= testX.shape[0]

	testingEnd = timer.time()

	testingTime = testingEnd-testingStart

	return [successRate, confusionMatrix, trainingTime, testingTime]

[Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z('working_dir/reviews.txt', 'working_dir/titles.txt', 'working_dir/ratings.txt', 'working_dir/bag_of_words_reviews.npy', 'working_dir/bag_of_words_titles.npy', 'word2vec_stem.txt')

[Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

[sR,cM,trT,teT] = get_performances(Xreview_train,Xreview_test,Y_train,Y_test,Z)

print(sR)
print(cM)
print(trT)
print(teT)
