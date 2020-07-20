import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):

	def __init__(self, vocab_size, embedding_dimension):

		super(SkipGram, self).__init__()


		self.embeddings = nn.Embedding(vocab_size, embedding_dimension)   
		self.context_embeddings = nn.Embedding(vocab_size, embedding_dimension)
		initrange = (2.0 / (vocab_size + embedding_dimension)) ** 0.5  # Xavier init
		self.embeddings.weight.data.uniform_(-initrange, initrange)
		self.context_embeddings.weight.data.uniform_(-0, 0)
		self.embeddings.weight.requires_grad = True
		self.context_embeddings.weight.requires_grad = True
        
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

		self.decode1 = nn.Linear(embedding_dimension, 256)
		self.dropout12 = nn.Dropout(0.5)
		self.decode2 = nn.Linear(256, 512)
		self.dropout23 = nn.Dropout(0.5)
		self.decode3 = nn.Linear(512, 1024)
		self.dropout34 = nn.Dropout(0.5)
		self.decode4 = nn.Linear(1024, 2048)
		self.dropout45 = nn.Dropout(0.5)
		self.decode5 = nn.Linear(2048, image_dimension)

	def forward(self,emb):

		d = self.dropout12(F.relu(self.decode1(emb)))
		d = self.dropout23(F.relu(self.decode2(d)))
		d = self.dropout34(F.relu(self.decode3(d)))
		d = self.dropout45(F.relu(self.decode4(d)))
		d = F.relu(self.decode5(d))
		return d



class MultiTaskLossWrapper(nn.Module):
	def __init__(self, task_num = 2):

		super(MultiTaskLossWrapper, self).__init__()
		
		self.task_num = task_num
		self.log_vars = nn.Parameter(torch.zeros((task_num)))

	def forward(self, skip_gram_loss, pred_image, image, pred_meta, meta):

		image_reconstruction_loss = nn.MSELoss()
        
		loss0 = skip_gram_loss
		loss1 = image_reconstruction_loss(pred_image, image)

		precision0 = torch.exp(-self.log_vars[0])
		loss0 = precision0*loss0 + self.log_vars[0]
		precision1 = torch.exp(-self.log_vars[1])
		loss1 = precision1*loss1 + self.log_vars[1]
		return (loss1 + loss0), loss1







