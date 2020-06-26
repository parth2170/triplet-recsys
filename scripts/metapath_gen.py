import os
import random
import pickle
import time
import numpy as np 
from tqdm import tqdm

def rate_weighted(rate_list):
	ids = [id_ for id_, _ in rate_list]
	ratings = np.array([r for _,r in rate_list])
	probs = ratings/np.sum(ratings)
	choice = np.random.choice(ids, p = probs)
	return choice


def metapaths_gen(user_dict, prod_dict, dataset_name, numwalks = 20, walklength = 10):
	outfile = open('../saved/'+dataset_name+'_metapaths.txt', 'w')
	for user0 in tqdm(user_dict):
		for _ in range(numwalks):
			path = user0
			user = user0
			for _ in range(walklength):
				prod = rate_weighted(user_dict[user])
				path = path + ' ' + prod
				user = rate_weighted(prod_dict[prod])
				path = path + ' ' + user
			
			path = path + '\n'
			outfile.write(path)
	outfile.close()

def main(dataset_name):
	user_dict = pickle.load(open('../saved/'+dataset_name+'_user_dict.pkl', 'rb'))
	prod_dict = pickle.load(open('../saved/'+dataset_name+'_prod_dict.pkl', 'rb'))
	start = time.time()
	metapaths_gen(user_dict, prod_dict, 'Baby')
	end = time.time()
	elapsed = end - start
	print('Time Taken = ', elapsed)
	with open('../log/'+dataset_name+'_reading_log.txt', 'a') as logfile:
		logfile.write('\nTime taken for generating metapaths = {:.4f} sec\n'.format(elapsed))

if __name__ == '__main__':
	main('Baby')