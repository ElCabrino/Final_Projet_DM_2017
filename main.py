from data_utils import *

reviews_path = 'working_dir/reviews.txt'
bag_of_words_path = 'working_dir/bag_of_words.npy'
ratings_path = 'working_dir/ratings.txt'
word2vec_path = 'word2vec.txt'
word2vec_stem_path = 'word2vec_stem.txt'

if __name__ == '__main__':
    # reading options
    if '-rm_working_dir' in sys.argv:
        rm_working_dir_content()
    if '-rm_dict' in sys.argv:
    	rm_dict()
    
    [X, Y, Z] = generate_and_get_X_Y_Z(reviews_path, ratings_path, bag_of_words_path, word2vec_stem_path)
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    
    X_train, X_test, Y_train, Y_test = shuffle_split(X, Y, 0.75)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)