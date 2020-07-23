import os
import gc
import random
import pickle
import numpy as np 
import pandas as pd
from tqdm import tqdm

def parse(path):
	g = open(path, 'r')
	for l in g:
		yield eval(l)

def normalize(x):
	return ((x - np.min(x))/(np.max(x) - np.min(x)))

def get_image_data(category):

	print('Loading image features data')
	prod_images = pickle.load(open('../saved/{}_prod_images.pkl'.format(category), 'rb'))
	cold_images = pickle.load(open('../saved/{}_cold_images.pkl'.format(category), 'rb'))

	prod_images = {pid:normalize(prod_images[pid]) for pid in prod_images}
	cold_images = {pid:normalize(cold_images[pid]) for pid in cold_images}

	return prod_images, cold_images

def encode(category, weight):
	user_dict = pickle.load(open('../saved/{}_user_dict.pkl'.format(category), 'rb'))
	prod_dict = pickle.load(open('../saved/{}_prod_dict.pkl'.format(category), 'rb'))

	vocab = list(user_dict.keys())
	vocab.extend(list(prod_dict.keys()))

	encoded_vocab = {vocab[i]:i for i in range(len(vocab))}
	reverse_encoded_vocab = {i:vocab[i] for i in range(len(vocab))}
	
	encoded_data = []
	with open('../saved/{}_metapaths_weight-True_.txt'.format(category), 'r') as file:
		for line in file:
			line = line.split()
			encoded_data.extend([encoded_vocab[token] for token in line])

	return encoded_data, encoded_vocab, reverse_encoded_vocab, user_dict, prod_dict

batch_id = 1
data_index = 0
user_queue = []
prod_queue = []
train_mode = 'prod'

def gen(encoded_data, reverse_encoded_vocab, batch_size, p_u_ratio, user_dict, prod_dict):
	
	global user_queue
	global prod_queue
	global data_index
	global batch_id
	global train_mode

	window_size = 5
	hop_size = 3

	batch = []

	while((data_index < len(encoded_data) - window_size)):
		window = encoded_data[data_index : data_index + window_size]
		word = window[int(window_size/2)]
		context_words = [window[j] for j in range(window_size) if j != int(window_size/2)]
		samples = [[word, context] for context in context_words]
		data_index += hop_size

		if data_index >= (len(encoded_data) - window_size):
			batch_id = 1
			data_index = 0

		if batch_id % p_u_ratio == 0:
			train_mode = 'user'
		else:
			train_mode = 'prod'
		try:
			user_dict[reverse_encoded_vocab[word]]
			user_queue.extend(samples)
		except:
			prod_dict[reverse_encoded_vocab[word]]
			prod_queue.extend(samples)
		while((len(batch) < batch_size)):
			if train_mode == 'user':
				if len(user_queue) == 0:
					break
				batch.append(user_queue.pop(0))
			else:
				if len(prod_queue) == 0:
					break
				batch.append(prod_queue.pop(0))
		if len(batch) == batch_size:
			batch_id += 1
			yield batch, batch_id, train_mode
			batch = []

def generate_image_batch(batch, prod_images, reverse_encoded_vocab):
	prods = [reverse_encoded_vocab[sample[0]] for sample in batch]
	image_batch = np.array([prod_images[prod] for prod in prods])
	return image_batch

