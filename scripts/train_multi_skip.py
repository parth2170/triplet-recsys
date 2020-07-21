
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
from multi_task_models import *

def get_embeddings(model, reverse_encoded_vocab, epoch, batch_id):

	embedding_matrix = model.embeddings.weight.data.clone()
	if torch.cuda.is_available():
		embedding_matrix = embedding_matrix.cpu()
	embedding_matrix = embedding_matrix.numpy()

	final_layer_embeddings = {}

	for i in reverse_encoded_vocab:
		final_layer_embeddings[reverse_encoded_vocab[i]] = embedding_matrix[i]
	
	with open('../saved/amazon_embeddings_e-{}_b-{}_.pkl'.format(epoch, batch_id), 'wb') as file:
		pickle.dump(final_layer_embeddings, file)

def train(category, weight):

	prod_images, _ = get_image_data(category)
	encoded_data, encoded_vocab, reverse_encoded_vocab, user_dict, prod_dict = encode(category, weight)
	
	total_words = len(encoded_vocab)
	print('\nVocabulary size = ', total_words)

	## Declare embedding dimension ##
	embedding_dimension = 100
	#################################

	skip_gram_model = SkipGram(vocab_size = total_words, embedding_dimension = embedding_dimension)
	image_model = ImageDecoder(embedding_dimension = embedding_dimension, image_dimension = 4096)
	multi_task_model = MultiTaskLossWrapper(task_num = 2)


	if torch.cuda.is_available():
			print('!!GPU!!')
			skip_gram_model.cuda()
			image_model.cuda()
			multi_task_model.cuda()

	skip_optimizer = torch.optim.Adam(skip_gram_model.parameters(), lr = 0.01)
	image_optimizer = torch.optim.Adam(list(image_model.parameters()) + list(skip_gram_model.parameters()) + list(multi_task_model.parameters()), lr = 0.01)

	print('\nStart Training\n')

	epoch_num = 5
	batch_size = 512
	Kill = True

	p_u_ratio = int(len(prod_dict)/len(user_dict))
	print('#products/#users = ', p_u_ratio)

	epoch = 0
	for batch, batch_id, train_mode in gen(encoded_data, reverse_encoded_vocab, batch_size, p_u_ratio, user_dict, prod_dict):
		if batch_id == 2:
			print('Saving model and embeddings')
			torch.save(image_model.state_dict(), '../saved/image_model.e-{}_b-{}_'.format(epoch, int(batch_id)))
			torch.save(skip_gram_model.state_dict(), '../saved/skip_gram_model.e-{}_b-{}_'.format(epoch, int(batch_id)))
			get_embeddings(skip_gram_model, reverse_encoded_vocab, epoch, int(batch_id))
			epoch += 1

		start_b = time.time()
		batch = np.array(batch)
		words = Variable(torch.LongTensor(batch[:,0]))
		context_words = Variable(torch.LongTensor(batch[:,1]))

		retain = False
		if train_mode == 'prod':
			image_batch = generate_image_batch(batch, prod_images, reverse_encoded_vocab)
			image_batch = Variable(torch.FloatTensor(image_batch))
			retain = True

		if torch.cuda.is_available():
			words = words.cuda()
			context_words = context_words.cuda()
			if train_mode == 'prod':
				image_batch = image_batch.cuda()

		skip_optimizer.zero_grad()
		image_optimizer.zero_grad()

		skip_gram_loss, skip_gram_emb = skip_gram_model(words, context_words)

		if train_mode == 'user':
			skip_gram_loss.backward()
			skip_optimizer.step()

		if train_mode == 'prod':
			pred_image = image_model(skip_gram_emb)
			image_loss, only_image = multi_task_model(skip_gram_loss, pred_image, image_batch)

			########################
			image_loss.backward()
			image_optimizer.step()
			########################

		if batch_id % 100 == 0:
			end_b = time.time()
			print('epoch = {}\tbatch = {}\tskip_gram_loss = {:.5f}\t\timage_loss = {:.5f}\t\ttime = {:.2f}'.format(epoch, batch_id, skip_gram_loss, only_image, end_b - start_b))
			start_b = time.time()
		if batch_id % 3000 == 0:
			print('Saving model and embeddings')
			torch.save(image_model.state_dict(), '../saved/image_model.e-{}_b-{}_'.format(epoch, int(batch_id)))
			torch.save(skip_gram_model.state_dict(), '../saved/skip_gram_model.e-{}_b-{}_'.format(epoch, int(batch_id)))
			get_embeddings(skip_gram_model, reverse_encoded_vocab, epoch, int(batch_id))
		if epoch >= epoch_num:
			print("\nOptimization Finished")
			break
	return model

if __name__ == '__main__':
	model = train("Men", True)



