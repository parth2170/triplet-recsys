import os
import random
import pickle
import numpy as np 
import pandas as pd
from tqdm import tqdm

def preprocess(line, encoded_vocab):
	'''
	takes a line of text and - 
	(i)		converts to lower case
	(ii)	removes punctuation
	(iii)	removes line breaks
	(iV)	encodes words  
	'''
	line = line.lower()
	line = line.replace('<br />', '')
	for punct in '&?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
		line = line.replace(punct, ' ')

	enc = []
	missing = 0
	for word in line.split():
		try:
			enc.append(encoded_vocab[word])
		except Exception as e:
			missing += 1
	return enc, missing


def get_corpus():

	print('Preparing data')
	vocab_path = 'data/aclImdb/imdb.vocab'
	vocab = []
	with open(vocab_path, 'r') as file:
		for line in file:
			words = line[:-1].split('-')
			vocab.extend(words)
	vocab = list(set(vocab))
	encoded_vocab = {vocab[i]:i for i in range(len(vocab))}
	reverse_encoded_vocab = {i:vocab[i] for i in range(len(vocab))}
	glove_embeddings, encoded_vocab, reverse_encoded_vocab = get_glove_embeddings(encoded_vocab)

	try:
		corpus = np.load('saved/imdb_corpus.npy')
	except:
		corpus_path = 'data/aclImdb/train/unsup'
		files = os.listdir(corpus_path)
		corpus = []
		for file in tqdm(files):
			with open(os.path.join(corpus_path, file), 'r') as file:
				for line in file:
					line, missing = preprocess(line, encoded_vocab)
					corpus.extend(line)
	np.save('saved/imdb_corpus.npy', corpus)
	return corpus, encoded_vocab, reverse_encoded_vocab, glove_embeddings

def co_occourance_matrix(encoded_data, encoded_vocab):
	'''
	see details in the mittens paper
	'''
	try:
		co_mat = np.load('saved/imdb_co_mat.npy')
	except:
		print('Computing co-occourance matrix')
		co_mat = np.zeros((len(encoded_vocab), len(encoded_vocab)))
		occourance_window_size = 10
		for i in tqdm(range(len(encoded_data) - occourance_window_size)):
			window = encoded_data[i:i+occourance_window_size]
			centre_word = window[int(occourance_window_size/2)]
			context_words = [(window[j], np.abs(j-occourance_window_size)) for j in range(occourance_window_size) if j != int(occourance_window_size/2)]
			for word in context_words:
				co_mat[centre_word][word[0]] += 1/word[1]
		np.save('saved/imdb_co_mat.npy', co_mat)
	return co_mat

data_index = 0

def generate_samples(encoded_data, batch_size):
	window_size = 5
	hop_size = 3
	global data_index
	samples = []
	while((data_index < len(encoded_data) - window_size)):
		while((len(samples) < batch_size)):
			window = encoded_data[data_index : data_index + window_size]
			word = window[int(window_size/2)]
			context_words = [window[j] for j in range(window_size) if j != int(window_size/2)]
			samples.extend([[word, context] for context in context_words])
			data_index += hop_size
		yield samples

def get_glove_embeddings(encoded_vocab):
	print('Reading Glove Embeddings')
	glove = {}
	new_encoded_vocab = {}
	counter = 0
	path = 'data/glove.6B/glove.6B.300d.txt'
	with open(path, 'r') as file:
		for line in tqdm(file):
			word = line.split()[0]
			try:
				encoded_vocab[word]
				vec = [float(i) for i in line.split()[1:]]
				glove[word] = vec 
				new_encoded_vocab[word] = counter
				counter += 1
			except:
				pass
	new_reverse_encoded_vocab = {v: k for k, v in new_encoded_vocab.items()}
	return glove, new_encoded_vocab, new_reverse_encoded_vocab


if __name__ == '__main__':
	corpus, encoded_vocab, reverse_encoded_vocab, glove_embeddings = get_corpus()













