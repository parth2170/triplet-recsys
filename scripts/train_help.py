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

def get_meta_data(prod_images, cold_images, category):
	try:
		prod_meta_info = pickle.load(open('../saved/{}_prod_meta_dimred.pkl'.format(category), 'rb'))
		cold_meta_info = pickle.load(open('../saved/{}_cold_meta_dimred.pkl'.format(category), 'rb'))
	except:
		meta_file_path = '../raw_data/metaaaaaa'
		prod_meta_info = {}
		cold_meta_info = {}
		for D in tqdm(parse(meta_file_path)):
			pid = D['asin']
			try:
				prod_images[pid]
				prod_meta_info[pid] = list(set([att for L in D['categories'] for att in L if(att != dataset_name and att != 'Clothing, Shoes & Jewelry')]))
			except:
				pass
			try:
				cold_images[pid]
				cold_meta_info[pid] = list(set([att for L in D['categories'] for att in L if(att != dataset_name and att != 'Clothing, Shoes & Jewelry')]))
			except:
				pass
		print('# of products for which meta information is available = ', len(prod_meta_info))
		with open('../saved/{}_prod_meta_dimred.pkl'.format(category), 'wb') as file:
			pickle.dump(prod_meta_info, file)
		with open('../saved/{}_cold_meta_dimred.pkl'.format(category), 'wb') as file:
			pickle.dump(cold_meta_info, file)
	return prod_meta_info, cold_meta_info

def get_image_data(category):

	print('Loading image features data')
	prod_images = pickle.load(open('../saved/{}_prod_images.pkl', 'rb'))
	cold_images = pickle.load(open('../saved/{}_cold_images.pkl', 'rb'))
	prod_meta_info, cold_meta_info = get_meta_data(prod_images, cold_images, category)

	all_meta_labels = [label for label in prod_meta_info[pid] for pid in prod_meta_info]
	all_meta_labels.extend([label for label in cold_meta_info[pid] for pid in cold_meta_info])
	all_meta_labels = list(set(all_meta_labels))

	return prod_images, cold_images, prod_meta_info, cold_meta_info, all_meta_labels

def encode(category, weight):
	user_dict = pickle.load(open('../saved/{}_user_dict.pkl', 'rb'))
	prod_dict = pickle.load(open('../saved/{}_prod_dict.pkl', 'rb'))

	vocab = list(user_dict.keys())
	vocab.extend(list(prod_dict.keys()))

	encoded_vocab = {vocab[i]:i for i in range(len(vocab))}
	reverse_encoded_vocab = {i:vocab[i] for i in range(len(vocab))}

	del user_dict
	del prod_dict
	gc.collect()

	encoded_data = []
	with open('../saved/{}_metapaths_weight-{}_.txt'.format(category, weight), 'r') as file:
		for line in file:
			line = line.split()
			line[-1] = line[-1][:-1]
			print(line)
			encoded_data.extend(line)

	return encoded_data, encoded_vocab, reverse_encoded_vocab

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
		yield sample

		