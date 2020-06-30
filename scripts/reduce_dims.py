import keras
import pickle
import numpy as np 
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation


def get_model(output_dims):
	model = Sequential()

	model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform', input_dim = 4096))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(output_dims, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(4096, activation='relu', kernel_initializer='he_uniform'))

	model.compile(optimizer='adam', loss=keras.losses.kullback_leibler_divergence, metrics=['mse'])

	return model

def train(output_dims, X):
	model = get_model(output_dims)
	print(model.summary())
	model.fit(X, X, batch_size = 128, epochs = 20)
	return model

def get_embeddings(model, prod_images, cold_images):
	layer_name = 'dense_3'
	intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
	for pid in prod_images:
		prod_images[pid] = intermediate_layer_model.predict(np.array([prod_images[pid]]))[0,:]
	for pid in cold_images:
		cold_images[pid] = intermediate_layer_model.predict(np.array([cold_images[pid]]))[0,:]
	return prod_images, cold_images

def main(dataset_name):
	output_dims = 100

	prod_images = pickle.load(open('../saved/{}/{}_prod_images.pkl'.format(dataset_name, dataset_name), 'rb'))
	cold_images = pickle.load(open('../saved/{}/{}_cold_images.pkl'.format(dataset_name, dataset_name), 'rb'))

	X = [cold_images[pid] for pid in cold_images]
	X = np.array(X)
	
	model = train(output_dims, X)
	model.save('../saved/{}/{}_dim_reduce_autoencoder.pkl'.format(dataset_name, dataset_name))

	prod_images, cold_images = get_embeddings(model, prod_images,cold_images)

	with open('../saved/{}/{}_cold_images.pkl'.format(dataset_name, dataset_name), 'wb') as file:
		pickle.dump(cold_images, file)
	with open('../saved/{}/{}_prod_images.pkl'.format(dataset_name, dataset_name), 'wb') as file:
		pickle.dump(prod_images, file)



if __name__ == '__main__':
	main('Baby')
