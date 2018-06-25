# a little script that takes a TCAT tweet file as input, runs them through NLTK/VADER sentiment analysis
# and writes a new files that adds four columns on the right

# The VADER library is documented here:
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
# Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# change as required:
filename_labeled = 'tcat_trump_5510tweets_labeled.csv'
colname_labeled_text = 'text'
colname_labeled_label = 'label'

filename_tolabel = 'tcat_trump_5510tweets.csv'
colname_tolabel_text = 'text'


import csv
import nltk
from nltk.tokenize import word_tokenize

csvread_labeled = open(filename_labeled, newline='\n')
csvreader_labeled = csv.DictReader(csvread_labeled, delimiter=',', quotechar='"')

# populate dictionary from CSV
train=[]
for row in csvreader_labeled:
	train.append((row[colname_labeled_text].lower(),row[colname_labeled_label]))

# create the overall feature vector:
all_words = set(word for passage in train for word in word_tokenize(passage[0]))

# create a feature vector for each text passage
t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]

# train the classifier
classifier = nltk.NaiveBayesClassifier.train(t)
# classifier.show_most_informative_features()

# read the lines to label, classify and write to new file
csvread_tolabel = open(filename_tolabel, newline='\n')
csvreader_tolabel = csv.DictReader(csvread_tolabel, delimiter=',', quotechar='"')
rowcount = len(open(filename_tolabel).readlines())

colnames = csvreader_tolabel.fieldnames
colnames.extend(['label'])

csvwrite = open(filename_tolabel[:-4] + "_BAYES.csv",'w',newline='\n')
csvwriter = csv.DictWriter(csvwrite, fieldnames=colnames)
csvwriter.writeheader()

for row in csvreader_tolabel:
	line_features = {word: (word in word_tokenize(row[colname_tolabel_text].lower())) for word in all_words}
	row.update({'label':classifier.classify(line_features)})
	csvwriter.writerow(row)
	rowcount -= 1
	print(rowcount)