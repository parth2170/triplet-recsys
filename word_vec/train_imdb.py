import os
import gc
import time
import torch
import random
import pickle
import numpy as np 
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
from make_imdb import *
from pytorch_model import Triplet_Embeddings

def get_embeddings(model, reverse_encoded_vocab):

	final_layer_embeddings = {}
	for i in reverse_encoded_vocab:
		ouput = model.pass_(torch.tensor([i]))[0]
		final_layer_embeddings[reverse_encoded_vocab[i]] = ouput.detach().numpy()
	with open('saved/final_layer_embeddings.pkl', 'wb') as file:
		pickle.dump(final_layer_embeddings, file)

def train():

	corpus, encoded_vocab, reverse_encoded_vocab, all_embeddings = get_corpus()

	total_words = len(all_embeddings)
	print('\nVocabulary size = ', total_words)


	co_mat = co_occourance_matrix(corpus, encoded_vocab)

	## Remove some embeddings
	random.seed(42)
	known_words = random.sample(list(all_embeddings.keys()), int(len(all_embeddings)/2))

	known_embeddings = {}
	for word in known_words:
		try:
			known_embeddings[word] = all_embeddings[word]
		except:
			pass	

	del all_embeddings
	gc.collect()

	## Declare model and optimizer
	model = Triplet_Embeddings(vocab_size = total_words, embedding_dimension = 300, output_dimension = 50, reverse_encoded_vocab = reverse_encoded_vocab, known_embeddings = known_embeddings)
	
	del known_embeddings
	gc.collect()

	if torch.cuda.is_available():
			print('!!GPU!!')
			model.cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

	print('\nStart Training\n')
	epoch_num = 5
	batch_size = 128
	Kill = True

	for epoch in range(epoch_num):
		start = time.time()
		batch_id = 0

		for batch in generate_triplets(corpus, co_mat, batch_size):
			
			batch = np.array(batch)

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
				for known in known_words:
					i = encoded_vocab[known]
					first_layer.grad[i] *= 0
			optimizer.step()
			batch_id += 1

			if batch_id % 50 == 0:
				print('epoch = {} batch = {} loss = {:.4f}'.format(epoch, batch_id, loss))

		end = time.time()

		print('####################### EPOCH DONE ##########################')
		print('epoch = {} batch = {} loss = {:.4f} time = {:.4f}'.format(epoch, batch_id, loss, end - start))
		torch.save(model.state_dict(), 'saved/imdb_model.epoch_{}_'.format(epoch))

	print("\nOptimization Finished")

	get_embeddings(model, reverse_encoded_vocab)
	return model

if __name__ == '__main__':
	model = train()

