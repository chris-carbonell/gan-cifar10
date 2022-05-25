# Dependencies

# general
import os
from pathlib import Path
import ssl  # https://stackoverflow.com/questions/69687794/unable-to-manually-load-cifar10-dataset

# data
import numpy as np

# viz
import matplotlib.pyplot as plt

# ml
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
# from keras.optimizers import Adam,SGD
from tensorflow.keras.optimizers import Adam, SGD  # https://stackoverflow.com/questions/62707558/importerror-cannot-import-name-adam-from-keras-optimizers

# Constants

image_shape = (32, 32, 3)

latent_dimensions = 100

PATH_OUTPUT = "./output/"
Path(PATH_OUTPUT).mkdir(parents=True, exist_ok=True)

PATH_MODEL = "./model/"
Path(PATH_MODEL).mkdir(parents=True, exist_ok=True)

# https://towardsdatascience.com/generating-modern-arts-using-generative-adversarial-network-gan-on-spell-39f67f83c7b4
PREVIEW_ROWS = 4
PREVIEW_COLS = 7

# noise for generated images
FIXED_NOISE = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, latent_dimensions))

# training

num_epochs=15000
batch_size=32
display_interval=100 # how often to save

# Funcs

def build_generator():
    '''
    Defining a utility function to build the Generator
    '''

    model = Sequential()

    #Building the input layer
    model.add(Dense(128 * 8 * 8, activation="relu",
                    input_dim=latent_dimensions))
    model.add(Reshape((8, 8, 128)))
    
    model.add(UpSampling2D())
    
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    # model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    
    model.add(UpSampling2D())
    
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    # model.add(BatchNormalization(momentum=0.78))
    model.add(Activation("relu"))
    
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))


    #Generating the output image
    noise = Input(shape=(latent_dimensions,))
    image = model(noise)

    return Model(noise, image)

def build_discriminator():
    '''
    Defining a utility function to build the Discriminator
    '''

    #Building the convolutional layers
    #to classify whether an image is real or fake
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2,
                    input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    # model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    # model.add(BatchNormalization(momentum=0.82))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.25))
    model.add(Dropout(0.25))
    
    #Building the output layer
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    image = Input(shape=image_shape)
    validity = model(image)

    return Model(image, validity)

def display_images(epoch):
    '''
    Defining a utility function to display the generated images
    '''
    
    # r, c = 4,4
    # noise = np.random.normal(0, 1, (r * c,latent_dimensions))
    # generated_images = generator.predict(noise)
    r, c = PREVIEW_ROWS, PREVIEW_COLS
    generated_images = generator.predict(FIXED_NOISE)

    #Scaling the generated images
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(generated_images[count, :,:,])
            # print(generated_images[count, :,:,])  # testing
            axs[i,j].axis('off')
            count += 1
    
    # plt.show()
    fig.set_figheight(6)
    fig.set_figwidth(8)
    plt.savefig(os.path.join(PATH_OUTPUT, f"{str(epoch).zfill(5)}.png"))
    plt.close()

# https://medium.com/swlh/creating-people-that-never-existed-generative-adversarial-networks-2e1af7397ee9
def show_losses(losses):
    losses = np.array(losses)
    
    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(PATH_OUTPUT, "training_losses.png"))

if __name__ == "__main__":

	# get data

    # https://stackoverflow.com/questions/69687794/unable-to-manually-load-cifar10-dataset
	ssl._create_default_https_context = ssl._create_unverified_context

	#Loading the CIFAR10 data
	(X, y), (_, _) = keras.datasets.cifar10.load_data()

	#Selecting a single class images
	#The number was randomly chosen and any number
	#between 1 to 10 can be chosen
	X = X[y.flatten() == 8]

	# build GAN

	# Building and compiling the discriminator
	discriminator = build_discriminator()
	discriminator.compile(loss='binary_crossentropy',
	                    optimizer=Adam(0.0002,0.5),
	                    metrics=['accuracy'])

	#Making the Discriminator untrainable
	#so that the generator can learn from fixed gradient
	discriminator.trainable = False

	# Building the generator
	generator = build_generator()

	#Defining the input for the generator
	#and generating the images
	z = Input(shape=(latent_dimensions,))
	image = generator(z)


	#Checking the validity of the generated image
	valid = discriminator(image)

	#Defining the combined model of the Generator and the Discriminator
	combined_network = Model(z, valid)
	combined_network.compile(loss='binary_crossentropy',
	                        optimizer=Adam(0.0002,0.5))

	# train

	losses=[]

	#Normalizing the input
	X = (X / 127.5) - 1.

	#Defining the Adversarial ground truths
	valid = np.ones((batch_size, 1))

	#Adding some noise
	valid += 0.05 * np.random.random(valid.shape)
	fake = np.zeros((batch_size, 1))
	fake += 0.05 * np.random.random(fake.shape)

	for epoch in range(num_epochs):
	            
	    #Training the Discriminator

	    #Sampling a random half of images
	    index = np.random.randint(0, X.shape[0], batch_size)
	    images = X[index]

	    #Sampling noise and generating a batch of new images
	    noise = np.random.normal(0, 1, (batch_size, latent_dimensions))
	    generated_images = generator.predict(noise)


	    #Training the discriminator to detect more accurately
	    #whether a generated image is real or fake
	    discm_loss_real = discriminator.train_on_batch(images, valid)
	    discm_loss_fake = discriminator.train_on_batch(generated_images, fake)
	    discm_loss = 0.5 * np.add(discm_loss_real, discm_loss_fake)

	    #Training the Generator

	    #Training the generator to generate images
	    #which pass the authenticity test
	    genr_loss = combined_network.train_on_batch(noise, valid)

	    #Tracking the progress
	    
	    if epoch % 5 == 0:
	        print(f"{str(epoch).zfill(5)}: [D loss: {discm_loss[0]}] [G loss: {genr_loss}]")
	        
	    if epoch % display_interval == 0:
	        clear_output(wait=True)
	        display_images(epoch)
	        
	    if epoch % 1000==0:
	        losses.append((discm_loss[0],genr_loss))	

    # save

    generator.save(os.path.join(PATH_MODEL, "generator.h5"))

    # evaluate

    show_losses(losses)

    #Plotting some of the original images
	s=X[:40]
	s = 0.5 * s + 0.5
	f, ax = plt.subplots(5,8, figsize=(16,10))
	for i, image in enumerate(s):
	    ax[i//8, i%8].imshow(image)
	    ax[i//8, i%8].axis('off')
	        
	plt.show()
	plt.savefig(os.path.join(PATH_OUTPUT, "original_images.png"))

	#Plotting some of the last batch of generated images
	noise = np.random.normal(size=(40, latent_dimensions))
	generated_images = generator.predict(noise)
	generated_images = 0.5 * generated_images + 0.5
	f, ax = plt.subplots(5,8, figsize=(16,10))
	for i, image in enumerate(generated_images):
	    ax[i//8, i%8].imshow(image)
	    ax[i//8, i%8].axis('off')
	        
	# plt.show()
	plt.savefig(os.path.join(PATH_OUTPUT, "last_batch_generated_images.png"))