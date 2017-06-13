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
    """
    # construction de X (a refaire)
    read_format_all_reviews(reviews_path)
    reviews_to_bag_of_words(reviews_path, bag_of_words_path)
    X = get_X(bag_of_words_path)
    # construction de Y:
    read_format_all_ratings(ratings_path)
    Y = get_Y(ratings_path)
    print(Y)

    Z = get_Z(word2vec_path)
    print(Z)
    

    read_format_all_reviews(reviews_path)
    create_stem_dict('dictionnary.txt', reviews_path)
    reviews_to_bag_of_words(reviews_path)
    X = get_X()
    print(X.sum())
    """
    """
    [X, Y, Z] = generate_and_get_X_Y_Z(reviews_path, ratings_path, bag_of_words_path, word2vec_stem_path)
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    """
    read_format_all_reviews(reviews_path)
    create_stem_dict('dictionnary.txt', reviews_path)
    read_format_review('datasets/reviews_tampax.csv', 'working_dir/tmpmiam.txt')
    read_format_rating('datasets/reviews_tampax.csv', 'working_dir/tmpmiam2.txt')
    count=0
    with open('working_dir/tmpmiam.txt', 'r') as f:
        for line in f:
            count += 1

    count2=0
    with open('working_dir/tmpmiam2.txt', 'r') as f:
        for line in f:
            count2 += 1

    print(count)
    print(count2)