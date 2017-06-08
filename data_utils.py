import csv
import string
import os
import shutil

def rm_working_dir_content():
	shutil.rmtree('working_dir/')
	os.makedirs('working_dir')

def rm_stopwords(user_input, stop_words):
    return list(set(user_input) - set(stop_words))

"""general function that format a review"""
def read_format_review(from_path, to_path, title=False):
	#I use this to remove punctuation
	translator=str.maketrans('','',string.punctuation)
	file = open(to_path+'.tmp',"w+")
	stop = []
	#making the list of stop words
	for line in open('stop_words.txt', 'r'):
		stop+= [line[:-1]]
	#reading the csv, removing punctuations and stop words
	with open(from_path) as csvfile:
		reader =  csv.DictReader(csvfile, delimiter = "\t")
		for row in reader:
			if not row['review']==None:
				for word in row['review'].translate(translator).split():
					if word not in stop:
						file.write(word+" ")
				file.write('\n')
	#doing steemin
	os.system('./stemmer.pl < '+ to_path+'.tmp'+' > ' + to_path)
	os.remove(to_path+'.tmp')

"""function to read and format all the reviews of this tp"""
def read_format_all_reviews(to_path):
	files = ['datasets/reviews_always.csv','datasets/reviews_gillette.csv','datasets/reviews_oral-b.csv','datasets/reviews_pantene.csv','datasets/reviews_tampax.csv']
	reviews = ['working_dir/reviews1.txt', 'working_dir/reviews2.txt', 'working_dir/reviews3.txt', 'working_dir/reviews4.txt', 'working_dir/reviews5.txt']
	read_format_review(files[0], reviews[0])
	read_format_review(files[1], reviews[1])
	read_format_review(files[2], reviews[2])
	read_format_review(files[3], reviews[3])
	read_format_review(files[4], reviews[4])

	with open(to_path,'wb') as reviews_file:
	    for rev in reviews:
	        with open(rev,'rb') as revd:
	            shutil.copyfileobj(revd, reviews_file)
	        os.remove(rev)