import os
import time
import torch
import random
import pickle
import numpy as np 
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_model import Triplet_Embeddings

def encode(path_to_corpus, vocabulary):
	'''
	Encodes words by assigning an integer to each word
	Encodes the entire corpus 
	vocabulary : list : list of the words in the corpus  
	'''
	encoded_vocab = {vocabulary[i]:i for i in range(len(vocabulary))}
	reverse_encoded_vocab = {i:vocabulary[i] for i in range(len(vocabulary))}
	encoded_data = []
	with open(path_to_corpus, 'r') as file:
		for line in file:
			tokens = line.split()
			encoded_data.extend([encoded_vocab[i] for i in tokens])
	return np.array(encoded_data), encoded_vocab, reverse_encoded_vocab

def co_occourance_matrix(encoded_data, encoded_vocab):
	'''
	see details in the mittens paper
	'''
	print('Computing co-occourance matrix')
	co_mat = np.zeros((len(encoded_vocab), len(encoded_vocab)))

	occourance_window_size = 10
	for i in tqdm(range(len(encoded_data) - occourance_window_size)):
		window = encoded_data[i:i+occourance_window_size]
		centre_word = window[int(occourance_window_size/2)]
		context_words = [(window[j], np.abs(j-occourance_window_size)) for j in range(occourance_window_size) if j != int(occourance_window_size/2)]
		for word in context_words:
			co_mat[centre_word][word[0]] += 1/word[1]
	return co_mat

def generate_triplets(encoded_data, co_mat):
	print('Generating samples')
	window_size = 3
	k = int(co_mat.shape[0] * 0.1) 
	i = 0
	samples = []
	for i in tqdm(range(len(encoded_data) - window_size)):
		window = encoded_data[i:i+window_size]
		anchor = window[int(window_size/2)]
		positives = [window[j] for j in range(window_size) if j != int(window_size/2)]
		negatives = np.argpartition(co_mat[anchor], k)[:k]
		for positive in positives:
			negative = np.random.choice(negatives)
			samples.append([anchor, positive, negative])
	samples = np.array(samples)
	return samples

def get_embeddings(model, reverse_encoded_vocab):

	final_layer_embeddings = {}
	for i in reverse_encoded_vocab:
		ouput = model.pass_(torch.tensor([i]))[0]
		final_layer_embeddings[reverse_encoded_vocab[i]] = ouput.detach().numpy()
	with open('saved/final_layer_embeddings.pkl', 'wb') as file:
		pickle.dump(final_layer_embeddings, file)

def train():

	all_embeddings = pickle.load(open('saved/toy_all_embeddings.pkl', 'rb'))
	cc = pd.read_csv('data/countries_capitals.csv')
	cc = cc.apply(lambda x : x.astype(str).str.lower(), axis = 0)
	known_embeddings = {}
	for country in list(cc['country']):
		try:
			known_embeddings[country] = all_embeddings[country]
		except:
			pass	
	encoded_data, encoded_vocab, reverse_encoded_vocab = encode('saved/toy_corpus.txt', list(all_embeddings.keys()))
	co_mat = co_occourance_matrix(encoded_data, encoded_vocab)
	samples = generate_triplets(encoded_data, co_mat)

	## Declare model and optimizer
	model = Triplet_Embeddings(vocab_size = len(all_embeddings), embedding_dimension = 300, output_dimension = 50, reverse_encoded_vocab = reverse_encoded_vocab, known_embeddings = known_embeddings)
	if torch.cuda.is_available():
			print('!!GPU!!')
			model.cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

	print('Start Training')
	epoch_num = 5
	batch_size = 128
	Kill = True

	for epoch in range(epoch_num):
		start = time.time()
		batch_id = 0
		while(batch_id < len(samples) - batch_size):
			
			batch = samples[batch_id:batch_id + batch_size]
			anchors = Variable(torch.LongTensor(batch[:,0]))
			positives = Variable(torch.LongTensor(batch[:,1]))
			negatives = Variable(torch.LongTensor(batch[:,2]))

			if torch.cuda.is_available():
				anchors = pos_u.cuda()
				positives = pos_v.cuda()
				negatives = neg_v.cuda()

			optimizer.zero_grad()

			loss = model(anchors, positives, negatives)
			loss.backward()

			if Kill:
				first_layer = next(model.parameters())
				for known in known_embeddings:
					i = encoded_vocab[known]
					first_layer.grad[i] *= 0
			optimizer.step()
			batch_id += batch_size


		end = time.time()
		print('epoch = {} loss = {:.4f} time = {:.4f}'.format(epoch, loss, end - start))
	torch.save(model.state_dict(), 'saved/model.epoch_{}_'.format(epoch))
	print("\nOptimization Finished")

	get_embeddings(model, reverse_encoded_vocab)
	return model

if __name__ == '__main__':
	model = train()

