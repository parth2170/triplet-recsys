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
import yaml

with open("./config.yaml") as file:
	config = yaml.load(file)


def train(category, weight,fix_decoder=False):

	prod_images = get_image_data(category)
	encoded_data, encoded_vocab, reverse_encoded_vocab, user_dict, prod_dict = encode(category, weight)
	total_words = len(encoded_vocab)
	print('\nVocabulary size = ', total_words)

	## Declare embedding dimension ##
	embedding_dimension = config["embedding_dim"]
	################################


	model = AutoEncoder(embedding_dimension,4096)
	if (bool(config["ae_fix_decoder"])):
		print(bool(config["ae_fix_decoder"]))     
		decoder_path = '../saved/'+config["model_name"]+'_image_model.e-{}_b-{}_'.format(int(config["multi_train_epochs"]), 3000)
		model.decoder.load_state_dict(torch.load(decoder_path))
		for param in model.decoder.parameters():
			param.requires_grad = False


	if torch.cuda.is_available():
			print('!!GPU!!')
			model.cuda()

	config["ae_fix_decoder"]= not bool(config["ae_fix_decoder"])
	with open('./config.yaml', 'w') as file:
		documents = yaml.dump(config, file)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

	print('\nStart Training\n')

	epoch_num = config["ae_epochs"]
	batch_size = config["ae_batch_size"]
	Kill = True

	p_u_ratio = int(len(prod_dict)/len(user_dict))
	print('#products/#users = ', p_u_ratio)

	epoch = 0
	for batch, batch_id, train_mode in gen(encoded_data, reverse_encoded_vocab, batch_size, p_u_ratio, user_dict, prod_dict):

		start_b = time.time()
		if train_mode == 'prod':
			image_batch = generate_image_batch(batch, prod_images, reverse_encoded_vocab)
			image_batch = Variable(torch.FloatTensor(image_batch))

		if torch.cuda.is_available():
			if train_mode == 'prod':
				image_batch = image_batch.cuda()


		if train_mode == 'prod':
			optimizer.zero_grad()
			criterion = nn.MSELoss()
			pred_image = model(image_batch)
			image_loss = criterion(pred_image,image_batch)
			
			########################
			image_loss.backward()
			optimizer.step()
			########################

		if batch_id % 100 == 0:
			end_b = time.time()
			print('epoch = {}\tbatch = {}\tskip_gram_loss = {:.5f}\t\ttime = {:.2f}'.format(epoch, batch_id, image_loss, end_b - start_b))
			start_b = time.time()
		if batch_id % 4000 == 0:
			print('Saving model and embeddings')
			torch.save(model.decoder.state_dict(), '../saved/'+config["model_name"]+'_ae_decoder.e-{}'.format(epoch))
			torch.save(model.encoder.state_dict(), '../saved/'+config["model_name"]+'_ae_encoder.e-{}'.format(epoch))
		if  batch_id >= 9000:
			print("\nOptimization Finished")
			break

	return model

if __name__ == '__main__':
	model = train(config["category"], bool(config["weights"]),bool(config["ae_fix_decoder"]))