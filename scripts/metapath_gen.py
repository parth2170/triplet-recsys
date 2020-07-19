import os
import random
import pickle
import time
import numpy as np 
from tqdm import tqdm

def rate_weighted(rate_list, weight):
	ids = [id_ for id_, _ in rate_list]
	if weight:
		ratings = np.array([r for _,r in rate_list])
		probs = ratings/np.sum(ratings)
		choice = np.random.choice(ids, p = probs)
	else:
		choice = np.random.choice(ids)
	return choice


def metapaths_gen(user_dict, prod_dict, dataset_name, numwalks = 20, walklength = 10, weight = False):
	outfile = open('../saved/{}_metapaths_weight-{}_.txt'.format(dataset_name, weight), 'w')
	for user0 in tqdm(user_dict):
		for _ in range(numwalks):
			path = user0
			user = user0
			for _ in range(walklength):
				prod = rate_weighted(user_dict[user], weight)
				path = path + ' ' + prod
				user = rate_weighted(prod_dict[prod], weight)
				path = path + ' ' + user
			
			path = path + '\n'
			outfile.write(path)
	outfile.close()

def main(dataset_name, weight):
	user_dict = pickle.load(open('../saved/{}_user_dict.pkl'.format(dataset_name), 'rb'))
	prod_dict = pickle.load(open('../saved/{}_prod_dict.pkl'.format(dataset_name), 'rb'))
	start = time.time()
	metapaths_gen(user_dict, prod_dict, dataset_name, weight = weight)
	end = time.time()
	elapsed = end - start
	print('Time Taken = ', elapsed)
	with open('../log/'+dataset_name+'_log.txt', 'a') as logfile:
		logfile.write('\nTime taken for generating metapaths = {:.4f} sec\n'.format(elapsed))

if __name__ == '__main__':
	cats = ['Baby', 'Men', 'Women', 'Shoes']
	for category in cats:
		main(category, weight = True)
