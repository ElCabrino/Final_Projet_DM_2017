from data_utils import *

reviews_path = 'working_dir/reviews.txt'
bag_of_words_path = 'working_dir/bag_of_words.txt'

if __name__ == '__main__':
	#reading options
	if '-rm_working_dir' in sys.argv:
		rm_working_dir_content()

	read_format_all_reviews(reviews_path)
	reviews_to_bag_of_words(reviews_path, bag_of_words_path)
	size_X = get_X_size(bag_of_words_path)