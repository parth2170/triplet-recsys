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
		meta_file_path = '../raw_data/meta_Clothing_Shoes_and_Jewelry.json'
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
	prod_images = pickle.load(open('../saved/{}_prod_images.pkl'.format(category), 'rb'))
	cold_images = pickle.load(open('../saved/{}_cold_images.pkl'.format(category), 'rb'))
	prod_meta_info, cold_meta_info = get_meta_data(prod_images, cold_images, category)

	all_meta_labels = [label for pid in prod_meta_info for label in prod_meta_info[pid] ]
	all_meta_labels.extend([label for pid in cold_meta_info for label in cold_meta_info[pid] ])
	all_meta_labels = list(set(all_meta_labels))

	return prod_images, cold_images, prod_meta_info, cold_meta_info, all_meta_labels

def encode(category, weight):
	user_dict = pickle.load(open('../saved/{}_user_dict.pkl'.format(category), 'rb'))
	prod_dict = pickle.load(open('../saved/{}_prod_dict.pkl'.format(category), 'rb'))

	vocab = list(user_dict.keys())
	vocab.extend(list(prod_dict.keys()))

	encoded_vocab = {vocab[i]:i for i in range(len(vocab))}
	reverse_encoded_vocab = {i:vocab[i] for i in range(len(vocab))}
	
	return encoded_vocab, reverse_encoded_vocab, user_dict, prod_dict

def rate_weighted(rate_list, weight = True):
	ids = [id_ for id_, _ in rate_list]
	if weight:
		ratings = np.array([r for _,r in rate_list])
		probs = ratings/np.sum(ratings)
		choice = np.random.choice(ids, p = probs)
	else:
		choice = np.random.choice(ids)
	return choice

def user_graph_walk(user, user_dict, prod_dict, encoded_vocab):
	# degree 1 neighbours
	prods_1 = user_dict[user]
	samples = [[encoded_vocab[user], encoded_vocab[id_]] for id_, _ in prods_1]
	# degree 2 neighbours
	for _ in range(int(len(prods_1)/2)):
		sample_prod = rate_weighted(prods_1)
		sample_user = rate_weighted(prod_dict[sample_prod])
		samples.append([encoded_vocab[user], encoded_vocab[sample_user]])
		# degree 3 neighbours
		for _ in range(int(len(prods_1)/8)):
			prods_2 = user_dict[sample_user]
			sample_prod = rate_weighted(prods_2)
			samples.append([encoded_vocab[user], encoded_vocab[sample_prod]])
	return samples

def prod_graph_walk(prod, user_dict, prod_dict, encoded_vocab):
	# degree 1 neighbours
	users_1 = prod_dict[prod]
	samples = [[encoded_vocab[prod], encoded_vocab[id_]] for id_, _ in users_1]
	# degree 2 neighbours
	for _ in range(int(len(users_1)/2)):
		sample_user = rate_weighted(users_1)
		sample_prod = rate_weighted(user_dict[sample_user])
		samples.append([encoded_vocab[prod], encoded_vocab[sample_prod]])
		# degree 3 neighbours
		for _ in range(int(len(users_1)/8)):
			users_2 = prod_dict[sample_prod]
			sample_user = rate_weighted(users_2)
			samples.append([encoded_vocab[prod], encoded_vocab[sample_user]])
	return samples

user_queue = []
user_index = 0

def generate_user_samples(user_list, user_dict, prod_dict, encoded_vocab, batch_size):

	global user_queue
	global user_index
	
	batch = []

	while(user_index < len(user_list)):
		user_queue.extend(user_graph_walk(user_list[user_index], user_dict, prod_dict, encoded_vocab))
		while(len(batch) < batch_size):
			if len(user_queue) == 0:
				break
			batch.append(user_queue.pop(0))
		user_index += 1
		if len(batch) == batch_size:
			yield batch


prod_queue = []
prod_index = 0

def generate_prod_samples(prod_list, user_dict, prod_dict, encoded_vocab, batch_size):

	global prod_queue
	global prod_index

	batch = []

	while(prod_index < len(prod_list)):
		prod_queue.extend(prod_graph_walk(prod_list[prod_index], user_dict, prod_dict, encoded_vocab))
		while(len(batch) < batch_size):
			if len(prod_queue) == 0:
				break
			batch.append(prod_queue.pop(0))
			prod_index += 1
		if len(batch) == batch_size:
			yield batch

def generate_image_batch(batch, prod_images, prod_meta_info, reverse_encoded_vocab):
	print(prod_meta_info.keys())    
	prods = [reverse_encoded_vocab[sample[0]] for sample in batch]
	image_batch = np.array([prod_images[prod] for prod in prods])
	meta_batch = np.array([prod_meta_info[prod] for prod in prods])
	return image_batch, meta_batch
		