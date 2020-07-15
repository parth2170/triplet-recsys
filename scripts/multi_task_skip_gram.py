import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

	def __init__(self, vocab_size, embedding_dimension):

		super(SkipGram, self).__init__()


		self.embeddings = nn.Embedding(vocab_size, embedding_dimension, sparse=True)   
		self.context_embeddings = nn.Embedding(vocab_size, embedding_dimension, sparse=True)
		initrange = (2.0 / (vocab_size + embedding_dimension)) ** 0.5  # Xavier init
		self.embeddings.weight.data.uniform_(-initrange, initrange)
		self.context_embeddings.weight.data.uniform_(-0, 0)
        
	def forward(self, word, context):

		embed_u = self.embeddings(word)
		embed_v = self.context_embeddings(context)

		score  = torch.mul(embed_u, embed_v)
		score = torch.sum(score, dim=1)
		log_target = F.logsigmoid(score).squeeze()
		return -1*log_target.mean(), embed_u




class ImageDecoder(nn.Module):

	def __init__(self, embedding_dimension, image_dimension, meta_dimension):

		super(ImageDecoder, self).__init__()

		# Input Embeddings
# 		self.embeddings = nn.Embedding(vocab_size, embedding_dimension)
# 		self.embeddings.weight.data.copy_(torch.from_numpy(embedding_initialization))
		# Image Decoder
		self.decode1 = nn.Linear(embedding_dimension, 512)
		self.dropout12 = nn.Dropout(0.5)
		self.decode2 = nn.Linear(512, 1024)
		self.dropout23 = nn.Dropout(0.5)
		self.decode3 = nn.Linear(1024, image_dimension)
		
		# Meta Information Layer
		self.meta = nn.Linear(embedding_dimension, meta_dimension)

	def forward(self,emb):

# 		emb = self.embeddings(word)
		d = self.dropout12(F.relu(self.decode1(emb)))
		d = self.dropout23(F.relu(self.decode2(d)))
		d = F.relu(self.decode3(d))

		m = F.sigmoid(self.meta(emb))

		return d, m



class MultiTaskLossWrapper(nn.Module):
	def __init__(self, task_num):

		super(MultiTaskLossWrapper, self).__init__()
		
		self.task_num = task_num
		self.log_vars = nn.Parameter(torch.zeros((task_num)))

	def forward(self, skip_gram_loss, pred_image, image, pred_meta, meta):

		image_reconstruction_loss = nn.MSELoss()
		meta_loss = nn.BCEWithLogitsLoss()
        
		loss0 = skip_gram_loss
		loss1 = image_reconstruction_loss(pred_image, image)
		# loss2 = meta_loss(pred_meta, meta)

		precision0 = torch.exp(-self.log_vars[0])
		loss0 = precision0*loss0 + self.log_vars[0]
		precision1 = torch.exp(-self.log_vars[1])
		loss1 = precision1*loss1 + self.log_vars[1]
		# precision2 = torch.exp(-self.log_vars[2])
		# loss2 = precision2*loss2 + self.log_vars[2]
		# return loss0+loss1+loss2
		return loss1 + loss0







