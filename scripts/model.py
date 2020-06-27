import pickle
import keras
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Lambda, Layer, Input, Embedding


def build_network(vocab_length, embedding_matrix):
	embeddingsize = 128 
	network = Sequential()
	network.add(Embedding(input_dim = vocab_length, 
						output_dim = 4096,
						weights=[embedding_matrix],
						input_length=None))
	network.add(Dense(1024, activation='relu', 
						kernel_initializer='he_uniform'))
	network.add(Dropout(0.5))
	network.add(Dense(512, activation='relu', 
						kernel_initializer='he_uniform'))
	network.add(Dropout(0.5)) 
	network.add(Dense(embeddingsize, activation=None, 
						kernel_initializer='he_uniform'))
	network.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))

	return network

class LossLayer(Layer):
	def __init__(self, alpha, **kwargs):
		self.alpha = alpha
		super(LossLayer, self).__init__(**kwargs)

	def loss(self, inputs):
		anchor, positive = inputs
		return keras.losses.kullback_leibler_divergence(anchor, positive)[0]

	def call(self, inputs):
		loss = self.loss(inputs)
		self.add_loss(loss)
		return loss


def build_model(vocab_length, network, margin=0.2):
	anchor_input = Input((vocab_length,), name="anchor_input")
	positive_input = Input((vocab_length,), name="positive_input")
	# Generate the encodings (feature vectors) for the three images
	encoded_a = network(anchor_input)
	encoded_p = network(positive_input)

	#TripletLoss Layer
	loss_layer = LossLayer(alpha=margin, name='triplet_loss_layer')([encoded_a,encoded_p])
	# Connect the inputs with the outputs
	network_train = Model(inputs=[anchor_input,positive_input], outputs=loss_layer)
	# return the model
	return network_train
	
