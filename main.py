from data_utils import *

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
    
    [Xreview, Xtitle, Y, Z] = generate_and_get_Xreview_Xtitle_Y_Z(reviews_path, titles_path, ratings_path, bag_of_words_reviews_path, bag_of_words_titles_path, word2vec_stem_path)
    print(Xreview.shape)
    print(Xtitle.shape)
    print(Y.shape)
    print(Z.shape)
    print(np.sum(Xreview))
    print(np.sum(Xtitle))
    
    #print(Y)
    #Y_vect = Y_to_vect(Y)
    #print(Y_vect)
    """
    X_train, X_test, Y_train, Y_test = shuffle_split(X, Y, 0.75)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    """

    #rnn.train(X, Y)