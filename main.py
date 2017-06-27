import data_utils as du
import numpy as np
import sys

import trained_random_baseline as tr_rand_baseline
import random_baseline as rand_baseline
import naive_bayes_baseline as bayes_baseline
import majority_baseline
import average_vector
import naive_bayes
import normalised_vector
import modified_bayesian_spam_filter as spam_filter
import linear_regression

reviews_path = 'working_dir/reviews.txt'
titles_path = 'working_dir/titles.txt'
bag_of_words_reviews_path = 'working_dir/bag_of_words_reviews.npy'
bag_of_words_titles_path = 'working_dir/bag_of_words_titles.npy'
ratings_path = 'working_dir/ratings.txt'
word2vec_path = 'word2vec.txt'
word2vec_stem_path = 'word2vec_stem.txt'

if __name__ == '__main__':
    # reading options
    if '-rm_working_dir' in sys.argv:
        rm_working_dir_content()
    if '-rm_dict' in sys.argv:
    	rm_dict()

    [Xreview, Xtitle, Y, Z] = du.generate_and_get_Xreview_Xtitle_Y_Z(reviews_path, titles_path, ratings_path, bag_of_words_reviews_path, bag_of_words_titles_path, word2vec_stem_path)
    
    n = 20
    success_rates = np.zeros((n, 8)) #9 is the number of algorithm
    success_rates_avrg = np.zeros((8))
    tr_times = np.zeros((8))
    te_times = np.zeros((8))
    confusions_matrices = [np.zeros((5,5)) for i in range(9)]

    print('Training and testing ...')
    for i in range(n):
        print('Step ' + str(i) + '           ', end = '\r')
        #getting the matrices, at every loop the are differents (because of the shuffle)
        [Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test] = du.shuffle_split(Xreview, Xtitle, Y, 0.75)

        #declaring needed variable:
        cM = [np.zeros((5,5)) for i in range(9)]

        #the baseline models
        [sR1,cM[0],teT1] = rand_baseline.get_performances(Y_test)
        [sR2,cM[1],trT2,teT2] = tr_rand_baseline.get_performances(Y_train, Y_test)
        [sR3,cM[2],trT3,teT3] = majority_baseline.get_performances(Y_train, Y_test)
        [sR4,cM[3],trT4,teT4] = bayes_baseline.get_performances(Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test, 0)

        #the other models
        [sR5,cM[4],trT5,teT5] = spam_filter.get_performances(Xreview_train,Xreview_test,Xtitle_train,Xtitle_test,Y_train,Y_test,1,10,True)
        [sR6,cM[5],trT6,teT6] = average_vector.get_performances(Xreview_train,Xreview_test,Xtitle_train,Xtitle_test,Y_train,Y_test,Z,0)
        [sR7,cM[6],trT7,teT7] = normalised_vector.get_performances(Xreview_train,Xreview_test, Xtitle_train, Xtitle_test,Y_train,Y_test,Z,0)
        [sR8,cM[7],trT8,teT8] = linear_regression.get_performances(Xtitle_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test, Z, 0)

        #collecting the results
        success_rates[i, :] = np.array([sR1, sR2, sR3, sR4, sR5, sR6, sR7, sR8])
        success_rates_avrg[:] += np.array([sR1, sR2, sR3, sR4, sR5, sR6, sR7, sR8])
        tr_times[:] += np.array([0, trT2, trT3, trT4, trT5, trT6, trT7, trT8])
        te_times[:] += np.array([teT1, teT2, teT3, teT4, teT5, teT6, teT7, teT8])
        confusions_matrices = [confusions_matrices[i] + cM[i] for i in range(8)] 

    print('Done!')

    tr_times /= n
    te_times /= n
    success_rates_avrg /= n
    confusions_matrices = [el/n for el in confusions_matrices]

    print('Success rates average:')
    print(success_rates_avrg)
    print('Success rates:')
    print(success_rates)
    print('Confusion matrices:')
    print(confusions_matrices)
    print('Training times:')
    print(tr_times)
    print('Testing times:')
    print(te_times)


