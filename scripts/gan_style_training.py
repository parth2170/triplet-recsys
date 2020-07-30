
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
from gan_models import *
import yaml

with open("./config.yaml") as file:
	config = yaml.load(file)
    
class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean
        
	def __call__(self, tensor):
		return tensor + (torch.randn(tensor.size()).cuda()) * self.std + self.mean
    
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_embeddings(model, reverse_encoded_vocab, epoch, batch_id):

	embedding_matrix = model.embeddings.weight.data.clone()
	if torch.cuda.is_available():
		embedding_matrix = embedding_matrix.cpu()
	embedding_matrix = embedding_matrix.numpy()

	final_layer_embeddings = {}

	for i in reverse_encoded_vocab:
		final_layer_embeddings[reverse_encoded_vocab[i]] = embedding_matrix[i]
	
	with open('../embeddings_models/GAN_embeddings_'+config["model_name"]+'_e-{}_b-{}_.pkl'.format(epoch, batch_id), 'wb') as file:
		pickle.dump(final_layer_embeddings, file)

def train():
	category = config["category"]
	gaussian = AddGaussianNoise()
	prod_images = get_image_data(category)
	encoded_data, encoded_vocab, reverse_encoded_vocab, user_dict, prod_dict = encode(category,bool( config["weights"]))
	
	total_words = len(encoded_vocab)
	print('\nVocabulary size = ', total_words)

	## Declare embedding dimension ##
	embedding_dimension = config["embedding_dim"]
	#################################

	if (bool(config["load_saved_decoder"])):
		skip_gram_model = SkipGram2(len(encoded_vocab),config["embedding_dim"],encoded_vocab,prod_images,4096,
                                    '../saved/'+str(config["embedding_dim"])+'dim_ae_encoder')
		image_model = ImageDecoder(embedding_dimension = embedding_dimension, image_dimension = 4096)
		image_model.load_state_dict(torch.load('../saved/'+str(config["embedding_dim"])+'dim_ae_decoder'))
# 		multi_task_model = MultiTaskLossWrapper(task_num = 2)
	else:
		skip_gram_model = SkipGram(vocab_size = total_words, embedding_dimension = embedding_dimension)
		image_model = ImageDecoder(embedding_dimension = embedding_dimension, image_dimension = 4096)
# 		multi_task_model = MultiTaskLossWrapper(task_num = 2)
        
        
# 	generator = MultiTaskLossWrapperGenerator(task_num = 2)
	discriminator = Discriminator(latent_emb_size=4096)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
			print('!!GPU!!')
			skip_gram_model.cuda()
			image_model.cuda()
# 			generator.cuda()
			discriminator.cuda()

	skip_optimizer = torch.optim.Adam(skip_gram_model.parameters(), lr = 0.01)
	gen_optimizer = torch.optim.Adam(list(image_model.parameters()) + list(skip_gram_model.parameters()), lr = 0.01)
	disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.01)


	print('\nStart Training\n')

	epoch_num = config["multi_train_epochs"]
	batch_size =  config["multi_train_batch_size"]
	Kill = True

	p_u_ratio = int(len(prod_dict)/len(user_dict))
	print('#products/#users = ', p_u_ratio)

	criterion = nn.BCELoss()
	image_loss = nn.MSELoss(reduction = 'mean')
	real_label = 1
	fake_label = 0

	epoch = 0
	for batch, batch_id, train_mode in gen(encoded_data, reverse_encoded_vocab, batch_size, p_u_ratio, user_dict, prod_dict):
		if batch_id == 2:
			print('Saving model and embeddings')
			torch.save(image_model.state_dict(), '../saved/'+config["model_name"]+'image_model.e-{}_b-{}_'.format(epoch, int(batch_id)))
			torch.save(skip_gram_model.state_dict(), '../saved/'+config["model_name"]+'skip_gram_model.e-{}_b-{}_'.format(epoch, int(batch_id)))
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

		skip_gram_loss, skip_gram_emb = skip_gram_model(words, context_words)

		if train_mode == 'user':
			skip_gram_loss.backward()
			skip_optimizer.step()

		if train_mode == 'prod':
			if (bool(config["add_noise"])):  
				skip_gram_emb = gaussian(skip_gram_emb) 
			disc_optimizer.zero_grad()
			# first pass real image embeddings to disc
			label = torch.full((batch_size,), real_label, device=device)
			output = discriminator(image_batch).view(-1)
			errD_real = criterion(output, label)
			# backprop real error
			errD_real.backward()
			D_x = output.mean().item()

			# get predicted image from generator
			label.fill_(fake_label)
			pred_image = image_model(skip_gram_emb)
			# pass predicted image through discriminator
			output = discriminator(pred_image.detach()).view(-1)
			errD_fake = criterion(output, label)
			errD_fake.backward()
			D_G_z1 = output.mean().item()

			errD = errD_real + errD_fake
			disc_optimizer.step()

			gen_optimizer.zero_grad()
			# update generator
			label.fill_(real_label)
			output = discriminator(pred_image).view(-1)
			disc_err_for_gen = criterion(output, label)
			# add skip gram loss to generator in multitask fashion
			# multi_gen_loss, only_gen_loss = generator(skip_gram_loss, disc_err_for_gen)
			# multi_gen_loss.backward()
			imgloss = image_loss(pred_image, image_batch)
			gen_loss = disc_err_for_gen + imgloss * 1e3 + skip_gram_loss * 1e2
			gen_loss.backward()
			# D_G_z2 = output.mean().item()
			# Update G
			gen_optimizer.step()

		if batch_id % 100 == 0:
			end_b = time.time()
			print('epoch = {}\tbatch = {}\tskip_gram_loss = {:.5f}\t\tdisc_loss = {:.5f}\t\tgen_loss = {:.7f}\t\timage_loss = {:.4f}'.format(epoch, batch_id, skip_gram_loss, errD, disc_err_for_gen, imgloss))
			start_b = time.time()
		if batch_id % 3000 == 0:
			print('Saving model and embeddings')
			torch.save(image_model.state_dict(), '../saved/'+config["model_name"]+'_image_model.e-{}_b-{}_'.format(epoch, int(batch_id)))
			torch.save(skip_gram_model.state_dict(), '../saved/'+config["model_name"]+'_skip_gram_model.e-{}_b-{}_'.format(epoch, int(batch_id)))
			get_embeddings(skip_gram_model, reverse_encoded_vocab, epoch, int(batch_id))
		if epoch > epoch_num:
			print("\nOptimization Finished")
			break
	return image_model

if __name__ == '__main__':
	model = train()



