from data_utils import *

reviews_path = 'working_dir/reviews.txt'
bag_of_words_path = 'working_dir/bag_of_words.txt'
ratings_path = 'working_dir/ratings.txt'

if __name__ == '__main__':
	#reading options
	if '-rm_working_dir' in sys.argv:
		rm_working_dir_content()

	#construction de X (a refaire)
	read_format_all_reviews(reviews_path)
	reviews_to_bag_of_words(reviews_path, bag_of_words_path)
	X = get_X(bag_of_words_path)
	#construction de Y:
	read_format_all_ratings(ratings_path)
	Y = get_Y(ratings_path)
	print(Y)
