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
from torch.utils.data import DataLoader

with open("./config.yaml") as file:
	config = yaml.load(file)
     
        
def train(category, weight):

	prod_images = get_image_data(category)
	dataset = np.asarray(list(prod_images.values()))   
	dataloader = DataLoader(dataset, batch_size= config["ae_batch_size"], shuffle=True)


	## Declare embedding dimension ##
	embedding_dimension = config["embedding_dim"]
	################################


	model = AutoEncoder(embedding_dimension,4096)

	if torch.cuda.is_available():
			print('!!GPU!!')
			model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

	print('\nStart Training\n')

	epoch_num = config["ae_epochs"]
	batch_size = config["ae_batch_size"]
	Kill = True


	for epoch in range(int(config['ae_epochs'])):
		for batch_id, batch in enumerate(dataloader) :
			start_b = time.time()
			image_batch = Variable(batch.float())

			if torch.cuda.is_available():
				image_batch = image_batch.cuda()
			optimizer.zero_grad()
			criterion = nn.MSELoss()
			pred_image = model(image_batch)
			image_loss = criterion(pred_image,image_batch)
			
			image_loss.backward()
			optimizer.step()


			if batch_id % 30 == 0:
				end_b = time.time()
				print('epoch = {}\tbatch = {}\tloss = {:.5f}\t\ttime = {:.2f}'.format(epoch, batch_id, image_loss, end_b - start_b))
				start_b = time.time()
			if batch_id % 100 == 0:
				print('Saving model and embeddings')
				torch.save(model.decoder.state_dict(), '../saved/'+str(config["embedding_dim"])+'dim_ae_decoder')
				torch.save(model.encoder.state_dict(), '../saved/'+str(config["embedding_dim"])+'dim_ae_encoder')
	print("\nOptimization Finished")

	return model

if __name__ == '__main__':
	model = train(config["category"], bool(config["weights"]))
