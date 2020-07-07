import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SkipMod(nn.Module):

	def __init__(self, vocab_size, embedding_dimension, output_dimension, reverse_encoded_vocab, known_embeddings):

		'''
		Input embedding dimension = 300 (glove embeddings)
		Output embedding dimension = 50
		reverse_encoded_vocab : dictionary : {code:'word'}
		known_embeddings : dictionary : {'word':[embedding]}	# contains words for which embedding is known
		'''

		super(SkipMod, self).__init__()

		self.vocab_size = vocab_size
		self.embedding_dimension = embedding_dimension

		self.embedding_initialization = self.init_embeddings(reverse_encoded_vocab, known_embeddings)
		self.embeddings = nn.Embedding(vocab_size, embedding_dimension)
		self.embeddings.weight.data.copy_(torch.from_numpy(self.embedding_initialization))		

		self.output = nn.Linear(embedding_dimension, vocab_size)

		self.loss_fn = nn.CrossEntropyLoss()

	def init_embeddings(self, reverse_encoded_vocab, known_embeddings):
		embedding_matrix = np.random.normal(0,1,(len(reverse_encoded_vocab), self.embedding_dimension))
		for i in reverse_encoded_vocab:
			try:
				tmp = known_embeddings[reverse_encoded_vocab[i]]
				tmp = (tmp - np.min(tmp))/(np.max(tmp) - np.min(tmp))
				embedding_matrix[i] = tmp
			except:
				pass
		return embedding_matrix

	def forward(self, word, context):
		
		word_vec = self.embeddings(word)
		logits = self.output(word_vec)
		loss = self.loss_fn(logits, context)

		return loss












