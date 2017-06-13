import csv
import string
import os
import shutil
import re
import sys
import numpy as np

"""this function delete the content of the working_dir/ folder"""


def rm_working_dir_content():
    shutil.rmtree('working_dir/')
    os.makedirs('working_dir')
    open('working_dir/.gitkeep', 'a').close()

"""function that remove stop words in user_input found in stop_words"""


def rm_stopwords(user_input, stop_words):
    return list(set(user_input) - set(stop_words))

"""general function that format a review"""
# T0D0: version avec le title


def read_format_review(from_path, to_path, title=False):
    # I use this to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    file = open(to_path + '.tmp', "w+")
    stop = []
    # making the list of stop words
    for line in open('stop_words.txt', 'r'):
        stop += [line[:-1]]
    # reading the csv, removing punctuations and stop words
    with open(from_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            # T0D0: voir si ca pete pas le code
            if not row['review'] == None:
                # formating the review to remove some rare case
                row_formatted = rm_rare_case(row['review'])
                for word in row_formatted.translate(translator).split():
                    if word not in stop:
                        file.write(word + " ")
                file.write('\n')
    # doing steemin
    os.system('./stemmer.pl < ' + to_path + '.tmp' + ' > ' + to_path)
    # removing tmp file
    os.remove(to_path + '.tmp')

"""function to read and format all the reviews of this tp"""


def read_format_all_reviews(to_path):
    if os.path.exists(to_path):
        return
    files = ['datasets/reviews_always.csv', 'datasets/reviews_gillette.csv',
             'datasets/reviews_oral-b.csv', 'datasets/reviews_pantene.csv', 'datasets/reviews_tampax.csv']
    reviews = ['working_dir/reviews1.txt', 'working_dir/reviews2.txt',
               'working_dir/reviews3.txt', 'working_dir/reviews4.txt', 'working_dir/reviews5.txt']
    # reading all the reviews
    read_format_review(files[0], reviews[0])
    read_format_review(files[1], reviews[1])
    read_format_review(files[2], reviews[2])
    read_format_review(files[3], reviews[3])
    read_format_review(files[4], reviews[4])

    # concatenating in one single file
    with open(to_path, 'wb') as reviews_file:
        for rev in reviews:
            with open(rev, 'rb') as revd:
                shutil.copyfileobj(revd, reviews_file)
            os.remove(rev)

"""convert the reviews in reviews_path in a bag of word representation in to_path
	return the number of reviews"""


def reviews_to_bag_of_words(reviews_path, to_path):
    if os.path.exists(to_path):
        return
    bag_of_words = open(to_path, "w+")
    dictionnary = []
    occurences = []
    n_reviews = 0

    # we count the occurence of each word of the dictionnary
    with open(reviews_path, "r") as reviews:
        for rev in reviews:
            n_reviews += 1
            for word in rev.split():
                # print(word)
                # if we know this word we simply add one to the occurences
                if word in dictionnary:
                    i = dictionnary.index(word)
                    occurences[i] += 1
                # else we add to the dictionnary and we add one the de
                # occurences list
                else:
                    dictionnary.append(word)
                    occurences.append(1)

    dict_size = len(dictionnary)

    # we re-open it to do the bag of words algorithm
    with open(reviews_path, "r") as reviews:
        # i write the size of the matrix X that is constructed with this file
        bag_of_words.write(str(n_reviews) + " " + str(dict_size) + "\n")
        for rev in reviews:
            for word in rev.split():
                i = dictionnary.index(word)
                bag_of_words.write(str(occurences[i]) + " ")
            bag_of_words.write("\n")

"""function that takes a string as parameter and remove all the pattern that looks like
 {word}.{word} or {word},{word} ..."""


def rm_rare_case(review):
    review = re.sub("(\s)([\., \,, \:, \-, \_, \;]+)(?=[a-z])", ' ', review)
    review = re.sub("([\., \,, \:, \-, \_, \;]+)(?=[a-z])", ' ', review)
    return review


def get_X_size(bag_of_words_path):
    with open(bag_of_words_path, 'r') as f:
        size_X_str = f.readline().split()
        size_X = [int(i) for i in size_X_str]
    return size_X


def get_X(bag_of_words_path):
    [n, d] = get_X_size(bag_of_words_path)
    X = np.zeros((n, d))
    itLine = 0
    itCol = 0
    with open(bag_of_words_path, "r") as bag_of_words_f:
        # jumping the first line because it's the size of X
        bag_of_words_f.readline()
        for review in bag_of_words_f:
            for number in review.split():
                X[itLine, itCol] = int(number)
                itCol += 1
            itCol = 0
            itLine += 1

    return X

"""general function that format a rating"""
# T0D0: version avec le title


def read_format_rating(from_path, to_path):
    file = open(to_path, "w+")

    # reading the csv, removing punctuations and stop words
    with open(from_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            if not row['user_rating'] == None:
                score = row['user_rating']
                file.write(score)
                file.write('\n')

"""function to read and format all the rating of this tp"""


def read_format_all_ratings(to_path):
    if os.path.exists(to_path):
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
    size = 0
    with open(rating_path, "r") as file:
        for line in file:
            size += 1

    return size


def get_Y(rating_path):
    n = get_Y_size(rating_path)
    Y = np.zeros((n, 5))
    itLine = 0
    with open(rating_path, "r") as ratings:
        for line in ratings:
            print(line[0])
            Y[itLine, int(line[0]) - 1] = 1
            itLine += 1

    return Y


def get_Z_size(word2vec_path):
    with open(word2vec_path, 'r') as f:
        size_Z_str = f.readline().split()
        size_Z = int(size_Z_str[0])
    return size_Z


def get_Z(word2vec_path):
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
