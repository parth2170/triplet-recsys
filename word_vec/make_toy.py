import os
import random
import pickle
import numpy as np 
import pandas as pd
from tqdm import tqdm

def get_vec(countries, capitals):
	path = 'data/glove.6B/glove.6B.300d.txt'
	V = {}
	with open(path, 'r') as file:
		for line in tqdm(file):
			word = line.split()[0]
			vec = [float(i) for i in line.split()[1:]]
			if (word in countries) or (word in capitals):
				V[word] = vec 
	return V

def countries_capitals():
	cc = pd.read_csv('data/countries_capitals.csv')
	cc = cc.apply(lambda x : x.astype(str).str.lower(), axis = 0)
	return cc

def make_corpus(cc, V, countries, capitals):
	# Number of times a country occours in the corpus
	N = 20
	with open('saved/toy_corpus.txt', 'w') as file:
		for index, row in tqdm(cc.iterrows()):
			try:
				V[row['country']]
				V[row['capital']]
			except:
				continue
			i = N
			line = ''
			while(i > 0):
				line += row['country']
				line += ' '
				if np.random.random() < 0.25:
					line += random.choice(capitals)
				else:
					line += row['capital']
				line += ' '
				i -= 1
			line += '\n'
			file.write(line)

if __name__ == '__main__':
	cc = countries_capitals()
	countries = list(cc['country'])
	capitals = list(cc['capital'])
	V = get_vec(countries, capitals)
	with open('saved/toy_all_embeddings.pkl', 'wb') as file:
		pickle.dump(V, file)
	# print('Countries found :', (set(V.keys())&set(countries)))
	# print('Capitals found :', (set(V.keys())&set(capitals)))
	make_corpus(cc, V, list(set(V.keys())&set(countries)), list(set(V.keys())&set(capitals)))
