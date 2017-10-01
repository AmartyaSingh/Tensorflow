import nltk
from nltk.tokenize import word_tokenize
#tokens are created i.e. the whole sentence is broken into an array of strings.
from nltk.stem import WordNetLemmatizer
#stemming: removing the 'ing' from words.
import numpy as np 
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000	

def create_lexicon(pos, neg):
	lexicon = []
	for fi in [pos, neg]:
		with open(fi,'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
							 #^upto how many lines.	
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)


	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	#w_counts = {'the':34234, 'and':13223}

	l2 = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)
	#taking words that occur less than 1000 but greater than 50 times.
	return l2

def sample_handling(sample, lexicon, classification):
	featureset = []

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents(:hm_lines):
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append
