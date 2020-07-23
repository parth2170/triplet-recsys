
import os
import array
import pickle
import urllib
import random
import requests
import numpy as np 
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def parse(path):
	g = open(path, 'r')
	for l in g:
		yield eval(l)


def get_urls(meta_file_path, prod_dict, category):
	try:
		imUrls = pickle.load(open('../saved/{}_imUrls.pkl'.format(category), 'rb'))
	except:
		imUrls = {}
		for D in tqdm(parse(meta_file_path)):
			try:
				pid = D['asin']
				prod_dict[pid]

				# img = Image.open(requests.get(D['imUrl'], stream=True).raw)
				imUrls[pid] = D['imUrl']
			except:
				pass
		print('# urls found = ', len(imUrls))
		with open('../saved/{}_imUrls.pkl'.format(category), 'wb') as file:
			pickle.dump(imUrls, file)
	return imUrls

def get_images(meta_file_path, prod_dict, category):
	try:
		images = pickle.load(open('../saved/{}_images.pkl'.format(category), 'rb'))
	except:
		imUrls = get_urls(meta_file_path, prod_dict, category)
		images = {}
		for pid in tqdm((list(imUrls.keys()))[:1000]):
			try:
				img = Image.open(requests.get(imUrls[pid], stream=True).raw)
				images[pid] = img
			except:
				pass
		print('# images found = ', len(images))
		with open('../saved/{}_images.pkl'.format(category), 'wb') as file:
			pickle.dump(images, file)
	return images

def get_user_images(imUrls, user_dict):
	users = list(user_dict.keys())
	users.sort()
	prods = []
	for user in users[80:90]:
		tmp = user_dict[user]
		prods.extend([pid for pid, _ in tmp])
	prods = list(set(prods))
	images = {}
	for pid in tqdm(prods):
		try:
			img = Image.open(requests.get(imUrls[pid], stream=True).raw)
			images[pid] = img
		except:
			pass
	return images, users[80:90]

def visualize_users():
	imUrls = pickle.load(open('../saved/{}_imUrls.pkl'.format('Men'), 'rb'))
	user_dict = pickle.load(open('../saved/Men_user_dict.pkl', 'rb'))
	images, users = get_user_images(imUrls, user_dict)
	embeddings = pickle.load(open('../saved/amazon_embeddings_e-1_b-6000_.pkl', 'rb'))

	image_keys = list(images.keys())
	image_keys.sort()

	X = [embeddings[uid] for uid in users]
	X.extend([embeddings[pid] for pid in image_keys])
	# X_emb = PCA(n_components=2).fit_transform(X)
	X_emb = TSNE(perplexity=30, n_components=2, n_iter=2000, n_jobs = -1).fit_transform(X)
	X_users = X_emb[:len(users)]
	X_prods = X_emb[len(users):]

	fig, ax = plt.subplots(figsize = (40, 40))

	artists = []

	for i, uid in enumerate(users):
		xu, yu = X_users[i]
		colour = np.random.rand(3,)
		icon = Image.open('incon.png')
		img = OffsetImage(icon, zoom = 0.02)
		ab = AnnotationBbox(img, (xu, yu), xycoords='data', frameon=True, bboxprops =dict(edgecolor=colour, lw=5))
		artists.append(ax.add_artist(ab))
		for pid, _ in user_dict[uid]:
			xp, yp = X_prods[image_keys.index(pid)]
			img = OffsetImage(images[pid], zoom = 0.08)
			ab = AnnotationBbox(img, (xp, yp), xycoords='data', frameon=True, bboxprops =dict(edgecolor=colour, lw=5))
			artists.append(ax.add_artist(ab))
	ax.update_datalim(X_emb)
	ax.autoscale()
	plt.show()


def visualize_prods(category):
	
	embeddings = pickle.load(open('../saved/amazon_embeddings_e-1_b-6000_.pkl', 'rb'))
	images = pickle.load(open('../saved/Men_images.pkl', 'rb'))
	
	image_keys = list(images.keys())
	image_keys.sort()

	X = [embeddings[pid] for pid in images]
	X_emb = PCA(n_components=2).fit_transform(X)
	# X_emb = TSNE(perplexity=30, n_components=2, n_iter=2000, n_jobs = -1).fit_transform(X)

	fig, ax = plt.subplots(figsize = (40, 40))

	artists = []
	for i, pid in enumerate(image_keys):
		x0, y0 = X_emb[i]
		img = OffsetImage(images[pid], zoom = 0.08)
		ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
		artists.append(ax.add_artist(ab))
	ax.update_datalim(X_emb)
	ax.autoscale()
	plt.show()



if __name__ == '__main__':
	# prod_dict = pickle.load(open('../saved/Men_prod_dict.pkl', 'rb'))
	# get_images('../raw_data/meta_Clothing_Shoes_and_Jewelry.json', prod_dict, 'Men')
	visualize_users()



