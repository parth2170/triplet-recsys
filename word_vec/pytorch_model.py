import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Triplet_Embeddings(nn.Module):

	def __init__(self, vocab_size, embedding_dimension, output_dimension, reverse_encoded_vocab, known_embeddings):

		'''
		Input embedding dimension = 300 (glove embeddings)
		Output embedding dimension = 50
		reverse_encoded_vocab : dictionary : {code:'word'}
		known_embeddings : dictionary : {'word':[embedding]}	# contains words for which embedding is known
		'''

		super(Triplet_Embeddings, self).__init__()

		self.embedding_dimension = embedding_dimension

		self.embedding_initialization = self.init_embeddings(reverse_encoded_vocab, known_embeddings)
		self.embeddings = nn.Embedding(vocab_size, embedding_dimension)
		self.embeddings.weight.data.copy_(torch.from_numpy(self.embedding_initialization))

		self.dense1 = nn.Linear(embedding_dimension, 100)
		self.dropout1 = nn.Dropout(0.5)
		self.dense2 = nn.Linear(100, output_dimension)

	def init_embeddings(self, reverse_encoded_vocab, known_embeddings):
		embedding_matrix = np.random.uniform(0,1,(len(reverse_encoded_vocab), self.embedding_dimension))
		for i in reverse_encoded_vocab:
			try:
				tmp = known_embeddings[reverse_encoded_vocab[i]]
				tmp = (tmp - np.min(tmp))/(np.max(tmp) - np.min(tmp))
				embedding_matrix[i] = tmp
			except:
				pass
		return embedding_matrix

	def pass_(self, sample):
		x = self.embeddings(sample)
		x = self.dropout1(F.relu(self.dense1(x)))
		x = self.dense2(x)
		x = (x - x.min())/(x.max() - x.min())
		return x

	def forward(self, anchor, positive, negative):
		
		anchor = self.pass_(anchor)
		positive = self.pass_(positive)
		negative = self.pass_(negative)

		distance_positive = F.cosine_similarity(anchor, positive)
		distance_negative = F.cosine_similarity(anchor, negative)

		losses = F.relu(distance_positive - distance_negative + 0.5)
		return losses.sum()












