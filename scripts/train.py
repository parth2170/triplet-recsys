import os
import time
import random
import pandas as pd
from tqdm import tqdm
import numpy as np 
from tensorflow.keras.utils import to_categorical
from model import *

data_index = 0

def encode(path_to_corpus, user_dict, prod_dict):
	all_keys = list(user_dict.keys()) + list(prod_dict.keys())
	all_keys.sort()
	encoded_vocab = {all_keys[i]:i for i in range(len(all_keys))}
	reverse_encoded_vocab = {i:all_keys[i] for i in range(len(all_keys))}
	encoded_data = []
	with open(path_to_corpus, 'r') as file:
		for line in file:
			tokens = line.split()
			encoded_data.extend([encoded_vocab[i] for i in tokens])
	return np.array(encoded_data), encoded_vocab, reverse_encoded_vocab

def generate_samples(encoded_data, encoded_vocab, dataset_name, batch_size):
	window_size = 5
	hop_size = 3
	global data_index
	
	if (data_index < len(encoded_data) - window_size):
		samples = []
		while((len(samples) < batch_size) and (data_index < len(encoded_data) - window_size)):
			window = encoded_data[data_index : data_index+window_size]
			centre_word = window[int(window_size/2)]
			context_words = [w for index, w in enumerate(window) if index != int(window_size/2)]
			for word in context_words:
				samples.append([to_categorical(centre_word, num_classes = len(encoded_vocab)), to_categorical(word, num_classes = len(encoded_vocab))])
			data_index += hop_size
		samples = np.array(samples)
		yield [samples[:,0], samples[:,1]], 0

def normalize(T):
	T = np.array(T)
	T = (T - np.min(T))/(np.max(T) - np.min(T))
	return T

def make_emb_matrix(reverse_encoded_vocab, prod_images):
	check = 0
	embedding_matrix = np.random.normal(0,1,(len(reverse_encoded_vocab), 100))
	for i in reverse_encoded_vocab:
		try:
			embedding_matrix[i] = normalize(prod_images[reverse_encoded_vocab[i]])
			check += 1
		except:
			pass
	print('# Images updated in embedding_matrix = ', check)
	return embedding_matrix

def train(dataset_name):
	batch_size = 32

	user_dict = pickle.load(open('../saved/{}/{}_user_dict.pkl'.format(dataset_name, dataset_name), 'rb'))
	prod_dict = pickle.load(open('../saved/{}/{}_prod_dict.pkl'.format(dataset_name, dataset_name), 'rb')) 

	print('Encoding Data')
	encoded_data, encoded_vocab, reverse_encoded_vocab = encode('../saved/{}/{}_metapaths.txt'.format(dataset_name, dataset_name), user_dict, prod_dict)

	prod_images = pickle.load(open('../saved/{}/{}_prod_images.pkl'.format(dataset_name, dataset_name), 'rb'))
	embedding_matrix = make_emb_matrix(reverse_encoded_vocab, prod_images)

	network = build_network(vocab_length = len(encoded_vocab), embedding_matrix = embedding_matrix)
	model = build_model(vocab_length = len(encoded_vocab), network = network)
	model.compile(optimizer='adam')
	print(model.summary())

	model.fit_generator(generate_samples(encoded_data, encoded_vocab, dataset_name, batch_size = batch_size), steps_per_epoch = int((((len(encoded_data) - 5)/3) * 4)/batch_size), epochs = 3)

	get_embeddings(model, reverse_encoded_vocab, dataset_name)
	return model

def get_embeddings(model, reverse_encoded_vocab, dataset_name):

	tmp = model.get_layer('sequential_1')
	print('Sequential Model')
	print(tmp.summary())

	tmp = Model(inputs=tmp.get_layer('embedding_1').input, outputs=tmp.get_layer('lambda_1').output)

	final_layer_embeddings = {}

	for i in reverse_encoded_vocab:
		final_layer_embeddings[reverse_encoded_vocab[i]] = tmp.predict([i])
	with open('../embeddings/{}_final_layer_embeddings.pkl'.format(dataset_name), 'wb') as file:
		pickle.dump(final_layer_embeddings, file)

if __name__ == '__main__':
	model = train('Baby')

