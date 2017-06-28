import csv
import string
import os
import shutil
import re
import sys
import linecache
import numpy as np


def rm_working_dir_content():
    """
    this function delete the content of the working_dir/ folder
    """
    if os.path.exists('working_dir/'):
        shutil.rmtree('working_dir/')
    os.makedirs('working_dir')
    open('working_dir/.gitkeep', 'a').close()


def rm_dict():
    """
    this function delete the customs dictionnary and word2vec
    """
    if os.path.exists('dictionnary_final.txt'):
        os.remove('dictionnary_final.txt')
    if os.path.exists('word2vec_stem.txt'):
        os.remove('word2vec_stem.txt')


# T0D0: version avec le title
def read_format_review(from_path, to_path_review, to_path_title):
    """
    general function that format a review in a csv file
    parameters:
                    from_path: where the .csv is stored
                    to_path: where to store the formatted reviews
    """
    print('Formating and removing stopwords of ' + from_path + '...', end='')
    # I use this to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    file_review = open(to_path_review + '.tmp', "w+")
    file_title = open(to_path_title + '.tmp', "w+")
    stop = []
    # making the list of stop words
    for line in open('stop_words.txt', 'r'):
        stop += [line[:-1]]
    # reading the csv, removing punctuations and stop words
    with open(from_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        # writing in both file the corresponding column
        for row in reader:
            #removing the punctuation with the translator
            for word in row['review'].translate(translator).split():
                # removing stopwords
                if not word.lower() in stop:
                    file_review.write(word + " ")
            file_review.write(' \n')
            for word in row['title'].translate(translator).split():
                # removing stopwords
                if not word.lower() in stop:
                    file_title.write(word + " ")
            file_title.write(' \n')

    file_review.close()
    file_title.close()
    # doing steemin
    os.system('./stemmer.pl < ' + to_path_review +
              '.tmp' + ' > ' + to_path_review)
    os.system('./stemmer.pl < ' + to_path_title +
              '.tmp' + ' > ' + to_path_title)
    # removing tmp file
    os.remove(to_path_review + '.tmp')
    os.remove(to_path_title + '.tmp')
    print(' Done!')


def read_format_all_reviews(to_path_reviews, to_path_title):
    """
    function to read and format all the reviews of this tp
    parameters:
                    to_path: where to store 'reviews.txt'
    """

    print('Creating ' + to_path_reviews + ' and ' + to_path_title)
    if os.path.exists(to_path_reviews):
        print('Files are already generated.')
        return
    files = ['datasets/reviews_always.csv', 'datasets/reviews_gillette.csv',
             'datasets/reviews_oral-b.csv', 'datasets/reviews_pantene.csv', 'datasets/reviews_tampax.csv']
    reviews = ['working_dir/reviews1.txt', 'working_dir/reviews2.txt',
               'working_dir/reviews3.txt', 'working_dir/reviews4.txt', 'working_dir/reviews5.txt']
    titles = ['working_dir/titles1.txt', 'working_dir/titles2.txt',
              'working_dir/titles3.txt', 'working_dir/titles4.txt', 'working_dir/titles5.txt']
    # reading all the reviews
    read_format_review(files[0], reviews[0], titles[0])
    read_format_review(files[1], reviews[1], titles[1])
    read_format_review(files[2], reviews[2], titles[2])
    read_format_review(files[3], reviews[3], titles[3])
    read_format_review(files[4], reviews[4], titles[4])

    # concatenating in one single file reviews
    with open(to_path_reviews, 'wb') as reviews_file:
        for rev in reviews:
            with open(rev, 'rb') as revd:
                shutil.copyfileobj(revd, reviews_file)
            os.remove(rev)
    print(to_path_reviews + ' created!')
    # concatenating in one single file titles
    with open(to_path_title, 'wb') as titles_file:
        for t in titles:
            with open(t, 'rb') as td:
                shutil.copyfileobj(td, titles_file)
            os.remove(t)
    print(to_path_title + ' created!')


def rm_rare_case(review):
    """
    function that takes a string as parameter and remove all the pattern that looks like
    {word}.{word} or {word},{word} ...
    parameters:
                    review: review to take care of
    """
    review = re.sub("(\s)([\., \,, \:, \-, \_, \;]+)(?=[a-z])", ' ', review)
    review = re.sub("([\., \,, \:, \-, \_, \;]+)(?=[a-z])", ' ', review)
    return review


def get_X_size(bag_of_words_path):
    """
    get the size of X store at the first line of bag_of_words_path
    parameters:
                    bag_of_words_path: where the file is stored
    """
    with open(bag_of_words_path, 'r') as f:
        size_X_str = f.readline().split()
        size_X = [int(i) for i in size_X_str]
    return size_X


def reviews_to_bag_of_words(reviews_path, to_path):
    """
    generates the bag of words representation of the reviews that are in reviews.txt
    and saves the result in X_f.npy
    parameters:
                    reviews_path: path of the file reviews.txt
    """

    print('Creating bag of words file ...', end='')
    if os.path.exists(to_path):
        print('File already generated.')
        return
    d = int(linecache.getline('word2vec_stem.txt', 1).split()[0])
    n = 0
    # counting the number of reviews
    with open(reviews_path, 'r+') as reviews_f:
        for line in reviews_f:
            n += 1
    # the dictionnary
    dictionnary_final = []
    with open('dictionnary_final.txt', 'r+') as dict_f:
        for line in dict_f:
            word = line.split()[0]
            dictionnary_final.append(word)
    # filling X
    X = np.zeros((n, d))
    itLine = 0
    with open(reviews_path) as reviews_f:
        for line in reviews_f:
            for word in line.split():
                # we only take care of words that are in the dictionnary
                if word in dictionnary_final:
                    index = dictionnary_final.index(word)
                    X[itLine, index] += 1
            itLine += 1

    np.save(to_path, X)
    print(' Done!')
    return X


def get_X(bag_of_words_path):
    print('Loading X...', end='')
    X = np.load(bag_of_words_path)
    print(' Done!')
    return X

# T0D0: version avec le title


def read_format_rating(from_path, to_path):
    """
    general function that format a rating
    parameters:
                    from_path: where the csv is stored
                    to_path: where to store the result
    """
    file = open(to_path, "w+")

    # reading the csv, removing punctuations and stop words
    with open(from_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            if not row['user_rating'] == None:
                score = row['user_rating']
                file.write(score)
                file.write('\n')


def read_format_all_ratings(to_path):
    """
    function to read and format all the rating of this tp
    parameters:
                    to_path: where to store the result
    """
    print('Creating ' + to_path + '...', end='')
    if os.path.exists(to_path):
        print(' ' + to_path + ' is already generated.')
        return
    files = ['datasets/reviews_always.csv', 'datasets/reviews_gillette.csv',
             'datasets/reviews_oral-b.csv', 'datasets/reviews_pantene.csv', 'datasets/reviews_tampax.csv']
    ratings = ['working_dir/ratings1.txt', 'working_dir/ratings2.txt',
               'working_dir/ratings3.txt', 'working_dir/ratings4.txt', 'working_dir/ratings5.txt']
    # reading all the reviews
    read_format_rating(files[0], ratings[0])
    read_format_rating(files[1], ratings[1])
    read_format_rating(files[2], ratings[2])
    read_format_rating(files[3], ratings[3])
    read_format_rating(files[4], ratings[4])

    # concatenating in one single file
    with open(to_path, 'wb') as ratings_file:
        for rate in ratings:
            with open(rate, 'rb') as rated:
                shutil.copyfileobj(rated, ratings_file)
            os.remove(rate)


def get_Y_size(rating_path):
    """
    return the number of scores
    """
    size = 0
    with open(rating_path, "r") as file:
        for line in file:
            size += 1

    return size


def get_Y(rating_path):
    """
    Get the matrix Y
    """
    n = get_Y_size(rating_path)
    Y = np.zeros((n, 5))
    itLine = 0
    with open(rating_path, "r") as ratings:
        for line in ratings:
            Y[itLine, int(line[0]) - 1] = 1
            itLine += 1

    return Y


def get_Z_size(word2vec_path):
    """
    get the number of words in Z
    """
    with open(word2vec_path, 'r') as f:
        size_Z_str = f.readline().split()
        size_Z = int(size_Z_str[0])
    return size_Z


def get_Z(word2vec_path):
    """
    get the matrix Z where the lines are the differentes words and the columns are
    the values of the word2vec representation
    """
    d = get_Z_size(word2vec_path)
    Z = np.zeros((d, 200))
    itLine = 0
    itCol = 0
    with open(word2vec_path, 'r') as word2vec:
        word2vec.readline()
        for line in word2vec:
            # print(line.split()[1:])
            for val in line.split()[1:]:
                Z[itLine, itCol] = float(val)
                itCol += 1
            itLine += 1
            itCol = 0
    return Z


def create_stem_dict(dictionnary_path, reviews_path, titles_path):
    """
    creates the custom dictionnary that is used to create X and Z 
    """
    if os.path.exists('word2vec_stem.txt') and os.path.exists('dictionnary_final.txt'):
        print('Dictionnary and custom word2vec are already generated.')
        return
        # doing steemin
    print('Doing stemming on the dictionnary...', end='')
    os.system('./stemmer.pl < ' + dictionnary_path +
              ' > ' + 'dictionnary_stem.tmp')
    print(' Done!')
    print('Removing duplicate words in the dictionnary...', end='')

    count = 1
    lines = 0
    f_stem_index = open('dictionnary_stem.txt', 'w+')
    unique_words = []
    # on enleve les doublons
    with open('dictionnary_stem.tmp', 'r+') as f_stem:
        size = f_stem.readline()
        for line in f_stem:
            word = line.split()[0]
            if not word in unique_words:
                f_stem_index.write(line[:-1] + ' ' + str(count) + '\n')
                unique_words.append(word)
                lines += 1
            count += 1
    os.remove('dictionnary_stem.tmp')
    print(' Done!')

    # on regarde tout les mots dans reviews.txt et titles.txt
    print('Removing words that are not in ' + reviews_path +
          ' and ' + titles_path + '...', end='')
    unique_words_reviews = []
    with open(reviews_path, 'r') as reviews_f:
        for line in reviews_f:
            for word in line.split():
                if not word in unique_words_reviews:
                    unique_words_reviews.append(word)
    with open(titles_path, 'r') as titles_f:
        for line in titles_f:
            for word in line.split():
                if not word in unique_words_reviews:
                    unique_words_reviews.append(word)

    f_stem_final = open('dictionnary_final.txt', 'w+')
    f_stem_index.seek(0, 0)
    size = 0
    for line in f_stem_index:
        word = line.split()[0]
        if word in unique_words_reviews:
            f_stem_final.write(line)
            size += 1
    print(' Done!')

    print('Mapping vectors from word2vec our dictionnary ...', end='')
    f_stem_final.seek(0, 0)
    nb_col = open('word2vec.txt', 'r').readline().split()[1]
    f_word2vec_stem = open('word2vec_stem.txt', 'w+')
    f_word2vec_stem.write(str(size) + ' ' + nb_col + '\n')
    for line in f_stem_final:
        index = int(line.split()[1]) + 1
        line_word2vec = linecache.getline('word2vec.txt', index)
        line_to_write = line.split()[0]
        for word in line_word2vec.split()[1:]:
            line_to_write += ' ' + word
        line_to_write += '\n'
        f_word2vec_stem.write(line_to_write)
    print(' Done!')
    os.remove('dictionnary_stem.txt')


def generate_and_get_Xreview_Xtitle_Y_Z(reviews_path, title_path, ratings_path, bag_of_words_reviews_path, bag_of_words_titles_path, word2vec_stem_path):
    """
    get the matrix Xreview, Xtitle, Y and Z
    """
    read_format_all_reviews(reviews_path, title_path)
    create_stem_dict('dictionnary.txt', reviews_path, title_path)
    # bag of words for the reviews
    reviews_to_bag_of_words(reviews_path, bag_of_words_reviews_path)
    # bag of words for the titles
    reviews_to_bag_of_words(title_path, bag_of_words_titles_path)
    read_format_all_ratings(ratings_path)
    Xreview = get_X(bag_of_words_reviews_path)
    Xtitle = get_X(bag_of_words_titles_path)
    Y = get_Y(ratings_path)
    Z = get_Z(word2vec_stem_path)

    return [Xreview, Xtitle, Y, Z]


def get_score(index, Y):
    """
    get the score in Y at the index 'index'
    """
    line = Y[index, :]
    count = 1
    for i in line:
        if i == 1:
            return count
        count += 1


def shuffle_split(Xreview, Xtitle, Y, ratio):
    """
    @param Xreview: bag of words matrix of all the reviews
    @param Xtitle: bag of words matrix of all the titles
    @param Y: score matrix
    @param ratio: between 0 and 1. Tells how to split the matrix. 0.75 means 75%, of the 2 matrices
    is for training and 25%, for the testing.
    
    return: the shuffled and splitted matrices
    """
    size_total = Xreview.shape[0]
    size_train = round(ratio * size_total)

    # shuffling the matrices
    randomize = np.arange(len(Xreview))
    np.random.shuffle(randomize)
    Xreview = Xreview[randomize, :]
    Xtitle = Xtitle[randomize, :]
    Y = Y[randomize, :]

    # splitting the matrices
    Xreview_train = Xreview[:size_train, :]
    Xreview_test = Xreview[size_train:, :]
    Xtitle_train = Xtitle[:size_train, :]
    Xtitle_test = Xtitle[size_train:, :]
    Y_train = Y[:size_train, :]
    Y_test = Y[size_train:, :]

    return [Xreview_train, Xreview_test, Xtitle_train, Xtitle_test, Y_train, Y_test]


def Y_to_vect(Y):
    """
    @param Y: the matrix to convert

    returns: a vector filled with the scores instead of a 1 at the corresponding column (matrix form)
    """
    vect_Y = np.zeros([Y.shape[0]])
    count = 0
    for line in Y:
        score = get_score(count, Y)
        vect_Y[count] = score
        count += 1
    return vect_Y
