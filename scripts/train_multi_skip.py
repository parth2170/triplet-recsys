
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
	encoded_vocab, reverse_encoded_vocab, user_dict, prod_dict = encode(category, weight)
	
	user_list = list(user_dict.keys())
	user_list.sort()
	prod_list = list(prod_dict.keys())
	prod_list.sort()

	total_words = len(encoded_vocab)
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

	p_u_ratio = int(len(prod_dict)/len(user_dict))
	print('#products/#users = ', p_u_ratio)

	for epoch in range(epoch_num):
		start_e = time.time()
		start_b = time.time()
		batch_id = 0

		flag = True
		while(flag):

			try:
				# First try to get user and product batches according to the p_u_ratio
				if batch_id % p_u_ratio == 0:
					batch = next(generate_user_samples(user_list, user_dict, prod_dict, encoded_vocab, batch_size))
					train_mode = 'user'
				else:
					batch = next(generate_prod_samples(prod_list, user_dict, prod_dict, encoded_vocab, batch_size))
					image_batch, meta_batch = generate_image_batch(batch, prod_images, prod_meta_info, reverse_encoded_vocab)
					train_mode = 'prod'
				# This will throw an error when user batches are finished
			except:
				# Try generating leftover image batches
				try:
					batch = next(generate_prod_samples(prod_list, user_dict, prod_dict, encoded_vocab, batch_size))
					image_batch, meta_batch = generate_image_batch(batch, prod_images, prod_meta_info, reverse_encoded_vocab)
					train_mode = 'prod'
				# If this throws error then images are finished
				except:
					end_e = time.time()
					print('####################### EPOCH DONE ##########################')
					print('epoch = {} batch = {} loss = {:.4f} time = {:.4f}'.format(epoch, batch_id, loss, end_e - start_b))
					print('Saving model and embeddings')
					torch.save(model.state_dict(), 'saved/imdb_model.e-{}_b-{}_'.format(epoch, int(batch_id/25000)))
					get_embeddings(model, reverse_encoded_vocab, epoch, int(batch_id/25000))
					flag = False

			batch = np.array(batch)

			words = Variable(torch.LongTensor(batch[:,0]))
			context_words = Variable(torch.LongTensor(batch[:,1]))

			if torch.cuda.is_available():
				words = words.cuda()
				context_words = context_words.cuda()
				if train_mode == 'prod':
					image_batch = image_batch.cuda()
					meta_batch = meta_batch.cuda()

			skip_optimizer.zero_grad()
			image_optimizer.zero_grad()

			skip_gram_loss, skip_gram_emb = skip_gram_model(words, context_words)

			if train_mode == 'user':
				skip_gram_loss.backward()
				skip_optimizer.step()
				skip_scheduler.step(skip_gram_loss)
				# Copy the skip-gram embedding matrix to the ImageDecoder 
# 				image_model.embeddings.weight.data.copy_(skip_gram_model.embeddings.weight.data)
				skip_optimizer.zero_grad()

			if train_mode == 'prod':
				pred_image, pred_meta = image_model(skip_gram_emb)
				skip_gram_emb_loss, skip_gram_image_loss = multi_task_model(skip_gram_loss, pred_image, image_batch, pred_meta, meta_batch)

				########################
				skip_gram_image_loss.backward()
				image_optimizer.step()
				image_scheduler.step(skip_gram_image_loss)
				image_optimizer.zero_grad()
				########################

				# Copy the ImageDecoder embedding matrix to skip-gram
				skip_gram_model.embeddings.weight.data.copy_(image_model.embeddings.weight.data)


			batch_id += 1

			if batch_id % 1000 == 0:
				end_b = time.time()
				print('epoch = {}\tbatch = {}\tloss = {:.4f}\ttime = {:.2f}'.format(epoch, batch_id, loss, end_b - start_b))
				start_b = time.time()
			if batch_id % 25000 == 0:
				print('Saving model and embeddings')
				torch.save(model.state_dict(), 'saved/imdb_model.e-{}_b-{}_'.format(epoch, int(batch_id/25000)))
				get_embeddings(model, reverse_encoded_vocab, epoch, int(batch_id/25000))

		print('epoch = {} batch = {} loss = {:.4f} time = {:.4f}'.format(epoch, batch_id, loss, end_e - start_b))
		print('Saving model and embeddings')
		torch.save(model.state_dict(), 'saved/imdb_model.e-{}_b-{}_'.format(epoch, int(batch_id/25000)))
		get_embeddings(model, reverse_encoded_vocab, epoch, int(batch_id/25000))

	print("\nOptimization Finished")
	return model

if __name__ == '__main__':
	model = train("Baby", True)

