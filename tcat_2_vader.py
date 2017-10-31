# a little script that takes a TCAT tweet file as input, runs them through NLTK/VADER sentiment analysis
# and writes a new files that adds four columns on the right

# The VADER library is documented here:
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
# Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# change as required:
filename = 'tcat_trump_5510tweets.csv'
textrowname = 'text'

import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

csvread = open(filename, newline='\n')
csvwrite = open(filename[:-4] + "_VADER.csv",'w',newline='\n')

csvreader = csv.DictReader(csvread, delimiter=',', quotechar='"')

fieldnames = csvreader.fieldnames
fieldnames.extend(['neg','neu','pos','compound'])

csvwriter = csv.DictWriter(csvwrite, fieldnames=fieldnames)
csvwriter.writeheader()

for row in csvreader:
	row.update(sid.polarity_scores(row[textrowname]))
	csvwriter.writerow(row)