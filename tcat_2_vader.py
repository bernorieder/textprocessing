# a little script that takes a TCAT tweet file as input, runs them through NLTK/VADER sentiment analysis
# and writes a new files that adds four columns on the right

filename = 'tcat_trump_5510tweets.csv'

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
	row.update(sid.polarity_scores(row["text"]))
	csvwriter.writerow(row)