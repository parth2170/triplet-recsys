'''
1. Read metadata to find category of each product
2. Read reviews file to get user, product graph and read images
'''

import time
import array
import pickle
import random
import numpy as np 
import pandas as pd
from tqdm import tqdm


def parse(path):
	g = open(path, 'r')
	for l in g:
		yield eval(l)

def readImageFeatures(path):
	f = open(path, 'rb')
	while True:
		asin = f.read(10)
		if asin == '':
			break
		a = array.array('f')
		a.fromfile(f, 4096)
		yield asin, a.tolist()

def reverse_(D):
	nD = {}
	for key in D:
		vals = D[key]
		for val, rate in vals:
			try:
				nD[val].append((key, rate))
			except:
				nD[val] = []
				nD[val].append((key, rate))
	for key in nD:
		nD[key] = list(set(nD[key]))
	return nD

def get_meta_data(meta_file_path, cats):
	try:
		prod_categories = pickle.load(open('../saved/prod_categories_meta.pkl', 'rb'))
	except:
		prod_categories = {cat:[] for cat in cats}
		for D in tqdm(parse(meta_file_path)):
			pid = D['asin']
			categories = set([att for L in D['categories'] for att in L])
			for tmp in list(categories.intersection(set(cats))):
				try:
					prod_categories[tmp].append(pid)
				except:
					pass
		prod_categories = {pid:k for k,v in prod_categories.items() for pid in v}
		with open('../saved/prod_categories_meta.pkl', 'wb') as file:
			pickle.dump(prod_categories, file)
	return prod_categories

def get_review_data(review_file_path, prod_categories, category):
	print('Reading reviews')
	data = []
	for D in tqdm(parse(review_file_path)):
		try:
			if category != prod_categories[D['asin']]:
				continue
			data.append({'uid':D['reviewerID'], 'pid':D['asin'], 'rating':D['overall']})
		except:
			pass
	return pd.DataFrame(data)

def get_image_data(train_prod_dict, test_prod_dict, cold_start, image_path, category, logfile):

	# Read image files
	cold_images = {}
	train_prod_images = {}
	test_prod_images = {}
	try:
		for pid, image in tqdm(readImageFeatures(image_path)):
			pid = pid.decode('ascii')
			try:
				tmp = train_prod_dict[pid]
				train_prod_images[pid] = image
			except:
				pass
			try:
				tmp = cold_start[pid]
				cold_images[pid] = image
			except:
				pass
			try:
				tmp = test_prod_dict[pid]
				test_prod_images[pid] = image
			except:
				pass
	except EOFError:
		print('File Read')
	
	with open('../saved/{}_cold_images.pkl'.format(category), 'wb') as file:
		pickle.dump(cold_images, file)
	with open('../saved/{}_train_prod_images.pkl'.format(category), 'wb') as file:
		pickle.dump(train_prod_images, file)
	with open('../saved/{}_test_prod_images.pkl'.format(category), 'wb') as file:
		pickle.dump(test_prod_images, file)

	return train_prod_images, test_prod_images, cold_images


def temp_read(review_file_path, prod_categories, category, logfile):
	# Read review file
	review_data = get_review_data(review_file_path, prod_categories, category)

	# Make user --> product dictionary
	user_dict_p = {u: list(p) for u, p in review_data.groupby('uid')['pid']}
	user_dict_r = {u: list(r) for u, r in review_data.groupby('uid')['rating']}
	logfile.write('Raw #users = {}\n'.format(len(user_dict_p)))
	user_dict = {}

	# Remove users with less than Q review
	Q = 9 if category == 'Women' else 5
	for user in user_dict_p:
		if len(user_dict_p[user]) < Q:
			continue
		user_dict[user] = list(zip(user_dict_p[user], user_dict_r[user]))

	user_dict[user] = list(zip(user_dict_p[user], user_dict_r[user]))
	logfile.write('#users with > 1 interactions = {}\n'.format(len(user_dict)))

	# Reverse user --> product dict to get product --> user dict
	prod_dict = reverse_(user_dict)
	logfile.write('#corresponding prods = {}\n'.format(len(prod_dict)))

	return user_dict, prod_dict

def Helper(cold_start, new_user_dict):
	n, tot = 0, 0
	for p in cold_start:
		for user, rating in cold_start[p]:
			tot += 1
			try:
				new_user_dict[user]
			except:
				n += 1
	return n, tot

def train_test_split(prod_dict, user_dict, logfile):
	# Keep one product for testing and rest for training for each user
	train_user_dict, test_user_dict = {}, {}
	random.seed(7)
	for user in user_dict:
		prods = user_dict[user]
		if len(prods) < 5:
			continue
		test_sample = random.choice(prods)
		test_user_dict[user] = [test_sample]
		prods.remove(test_sample)
		train_user_dict[user] = prods
	test_prod_dict = reverse_(test_user_dict)
	train_prod_dict = reverse_(train_user_dict)

	logfile.write('#users in train set = {}\n'.format(len(train_user_dict)))
	logfile.write('#users in test set = {}\n'.format(len(test_user_dict)))
	logfile.write('#products in train set = {}\n'.format(len(train_prod_dict)))
	logfile.write('#products in test set = {}\n'.format(len(test_prod_dict)))
	return train_user_dict, train_prod_dict, test_user_dict, test_prod_dict


def remove_cold_start(prod_dict, user_dict, logfile):
	random.seed(7)
	len_dict1 = {prod:prod_dict[prod] for prod in prod_dict if (len(prod_dict[prod]) < 25 and len(prod_dict[prod]) >= 10)}
	random.seed(7)
	len_dict2 = {prod:prod_dict[prod] for prod in prod_dict if (len(prod_dict[prod]) < 50 and len(prod_dict[prod]) >= 25)}
	keys1 = random.sample(list(len_dict1.keys()), 30)
	keys2 = random.sample(list(len_dict2.keys()), 20)
	cold_start, new_prod_dict = {}, {}
	for prod in prod_dict:
		if ((prod in keys1) or (prod in keys2)):
			cold_start[prod] = prod_dict[prod]
		else:
			new_prod_dict[prod] = prod_dict[prod]
	new_user_dict = reverse_(new_prod_dict)
	logfile.write('#cold start products = {}\n'.format(len(cold_start)))
	n, tot = Helper(cold_start, new_user_dict)
	logfile.write('#users in cold products = {}\n'.format((tot)))
	logfile.write('#users in cold products who are not present in user_dict = {}\n'.format((n)))
	logfile.write('#users after removing Cold Start Products = {}\n'.format(len(new_user_dict)))
	logfile.write('#products after removing Cold Start Products = {}\n'.format(len(new_prod_dict)))
	return cold_start, new_prod_dict, new_user_dict

def refine(user_dict, prod_dict, cold_start, prod_images, cold_images, dataset_name):
	ref_cold, ref_user, ref_prod = {}, {}, {}
	ref_cold = {pid:cold_start[pid] for pid in cold_images}
	ref_prod = {pid:prod_dict[pid] for pid in prod_images}
	ref_user = reverse_(ref_prod)
	return ref_cold, ref_user, ref_prod

def main(review_file_path, image_path, prod_categories, category):

	print('\n', category)
	logfile = open('../log/'+category+'_log.txt', 'w')

	start = time.time()

	# Read reviews
	user_dict, prod_dict = temp_read(review_file_path, prod_categories, category, logfile)

	# Test-Train Split
	train_user_dict, train_prod_dict, test_user_dict, test_prod_dict = train_test_split(prod_dict, user_dict, logfile)

	# Remove some products for cold start testing
	cold_start, train_prod_dict, train_user_dict = remove_cold_start(train_prod_dict, train_user_dict, logfile)

	# Read images
	train_prod_images, test_prod_images, cold_images = get_image_data(train_prod_dict, test_prod_dict, cold_start, image_path, category, logfile)

	# Remove products whose images are not present
	cold_start, train_user_dict, train_prod_dict = refine(train_user_dict, train_prod_dict, cold_start, train_prod_images, cold_images, category)

	with open('../saved/{}_cold_start.pkl'.format(category), 'wb') as file:
		pickle.dump(cold_start, file)
	with open('../saved/{}_train_user_dict.pkl'.format(category), 'wb') as file:
		pickle.dump(train_user_dict, file)
	with open('../saved/{}_train_prod_dict.pkl'.format(category), 'wb') as file:
		pickle.dump(train_prod_dict, file)
	with open('../saved/{}_test_user_dict.pkl'.format(category), 'wb') as file:
		pickle.dump(test_user_dict, file)
	with open('../saved/{}_test_prod_dict.pkl'.format(category), 'wb') as file:
		pickle.dump(test_prod_dict, file)
	
	end = time.time()
	elapsed = end - start

	logfile.write('\nFinal stats after removing missing image files:\n')
	logfile.write('#train users = {}\n #train products = {}\n #cold_start = {}\n'.format(len(train_user_dict), len(train_prod_dict), len(cold_start)))
	logfile.write('#test users = {}\n #test products = {}\n'.format(len(test_user_dict), len(test_prod_dict)))
	logfile.write('\nTime taken for reading data = {:.4f} sec\n'.format(elapsed))
	logfile.close()

if __name__ == '__main__':

	cats = ['Women', 'Men', 'Shoes']

	review_file_path = '../raw_data/reviews_Clothing_Shoes_and_Jewelry.json'
	image_path = '../raw_data/image_features_Clothing_Shoes_and_Jewelry.b'
	meta_file_path = '../raw_data/meta_Clothing_Shoes_and_Jewelry.json'


	prod_categories = get_meta_data('../raw_data/meta_Clothing_Shoes_and_Jewelry.json', cats)

	for category in cats:
		main(review_file_path, image_path, prod_categories, category)
