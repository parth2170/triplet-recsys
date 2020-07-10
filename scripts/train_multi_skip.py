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
from train_help import *
from multi_task_skip_gram import *

def get_embeddings(model, reverse_encoded_vocab, epoch, batch_id):

	embedding_matrix = model.embeddings.weight.data
	if torch.cuda.is_available():
		embedding_matrix = embedding_matrix.cpu()
	embedding_matrix = embedding_matrix.numpy()

	final_layer_embeddings = {}

	for i in reverse_encoded_vocab:
		final_layer_embeddings[reverse_encoded_vocab[i]] = embedding_matrix[i]
	
	with open('saved/imdb_embeddings_e-{}_b-{}_.pkl'.format(epoch, batch_id), 'wb') as file:
		pickle.dump(final_layer_embeddings, file)

def train(category, weight):

	prod_images, _, prod_meta_info, _, all_meta_labels = get_image_data(category)
	prod_meta_info, _ = get_meta_data(prod_images, cold_images, category)

	encoded_data, _, reverse_encoded_vocab = encode(category, weight)

	total_words = len(reverse_encoded_vocab)
	print('\nVocabulary size = ', total_words)

	## Declare embedding dimension ##
	embedding_dimension = 100
	#################################

	skip_gram_model = SkipGram(vocab_size = total_words, embedding_dimension = embedding_dimension)
	image_model = ImageDecoder(embedding_dimension = embedding_dimension, image_dimension = 4096, meta_dimension = len(all_meta_labels))
	multi_task_model = MultiTaskLossWrapper(task_num = 2)


	if torch.cuda.is_available():
			print('!!GPU!!')
			skip_gram_model.cuda()
			image_model.cuda()

	skip_optimizer = torch.optim.Adam(skip_gram_model.parameters(), lr = 0.01)
	skip_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(skip_optimizer, 'min', patience = 200, verbose = True)
	image_optimizer = torch.optim.Adam(image_model.parameters(), lr = 0.01)
	image_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(image_optimizer, 'min', patience = 200, verbose = True)

	print('\nStart Training\n')

	epoch_num = 2
	batch_size = 256
	Kill = True

	for epoch in range(epoch_num):
		start_e = time.time()
		start_b = time.time()
		batch_id = 0

		for batch in generate_samples(encoded_data, batch_size):
			
			batch = np.array(batch)

			words = Variable(torch.LongTensor(batch[:,0]))
			context_words = Variable(torch.LongTensor(batch[:,1]))

			if torch.cuda.is_available():
				words = words.cuda()
				context_words = context_words.cuda()

			optimizer.zero_grad()

			emb, skip_gram_logits, skip_gram_loss = skip_gram_model(words, context_words)
			pred_image, pred_meta = image_model(emb)
			multi_task_loss = multi_task_model(words, skip_gram_loss, pred_image, image, pred_meta, meta)

			# loss.backward()

			if Kill:
				first_layer = next(model.parameters())
				for known in known_words:
					i = encoded_vocab[known]
					first_layer.grad[i] *= 0

			optimizer.step()
			scheduler.step(loss)

			batch_id += 1

			if batch_id % 1000 == 0:
				end_b = time.time()
				print('epoch = {}\tbatch = {}\tloss = {:.4f}\ttime = {:.2f}'.format(epoch, batch_id, loss, end_b - start_b))
				start_b = time.time()
			if batch_id % 25000 == 0:
				print('Saving model and embeddings')
				torch.save(model.state_dict(), 'saved/imdb_model.e-{}_b-{}_'.format(epoch, batch_id/25000))
				get_embeddings(model, reverse_encoded_vocab, epoch, batch_id)

		end_e = time.time()

		print('####################### EPOCH DONE ##########################')
		print('epoch = {} batch = {} loss = {:.4f} time = {:.4f}'.format(epoch, batch_id, loss, end_e - start_b))

	print("\nOptimization Finished")
	return model

if __name__ == '__main__':
	model = train()

