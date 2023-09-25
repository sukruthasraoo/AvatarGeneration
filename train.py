import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from tqdm import tqdm
import cv2

latent_dim = 100 

def build_genera(latent_dim):
    genera = models.Sequential()

    genera.add(layers.Dense(128, input_dim=latent_dim))
    genera.add(layers.LeakyReLU(alpha=0.2))

    genera.add(layers.Dense(256))
    genera.add(layers.LeakyReLU(alpha=0.2))

    genera.add(layers.Dense(784, activation='tanh'))
    genera.add(layers.Reshape((28, 28, 1)))

    return genera

def build_discriminator(input_shape):
    discriminator = models.Sequential()

    discriminator.add(layers.Flatten(input_shape=input_shape))

    discriminator.add(layers.Dense(256))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    discriminator.add(layers.Dense(128))
    discriminator.add(layers.LeakyReLU(alpha=0.2))

    discriminator.add(layers.Dense(1, activation='sigmoid'))

    return discriminator

def build_gan(genera, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    x = genera(gan_input)
    gan_output = discriminator(x)
    gan = models.Model(gan_input, gan_output)
    return gan

discriminator = build_discriminator(input_shape=(28, 28, 1))
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

genera = build_genera(latent_dim)
discriminator.trainable = False

gan = build_gan(genera, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

epochs = 10000
batch_size = 128
sample_interval = 1000

dataset_directory = r'E:\sukrutha\face dataset\img_align_celeba\img_align_celeba'

preprocessed_image_paths = []

for filename in tqdm(os.listdir(dataset_directory)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(dataset_directory, filename)
        preprocessed_image_paths.append(img_path)

for epoch in range(epochs):
    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
       
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = genera.predict(noise)
        
        for i, generated_image in enumerate(generated_images):
            save_path = f'generated_image_epoch_{epoch}_sample_{i}.jpg'
            cv2.imwrite(save_path, generated_image * 255)  

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    labels_real = np.ones((batch_size, 1))
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = genera.predict(noise)
    labels_fake = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_images, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    labels_gan = np.ones((batch_size, 1))
    
    g_loss = gan.train_on_batch(noise, labels_gan)

    if epoch % sample_interval == 0:
        print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = genera.predict(noise)
        
        for i, generated_image in enumerate(generated_images):
            save_path = f'generated_image_epoch_{epoch}_sample_{i}.jpg'
            cv2.imwrite(save_path, generated_image * 255)  

import matplotlib.pyplot as plt

dloss = []
gloss = []

for epoch in range(epochs):
    dloss.append(d_loss[0])
    gloss.append(g_loss)

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), dloss, label="Discriminator Loss")
plt.plot(range(epochs), gloss, label="genera Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Loss Curves")
plt.show()

noise = np.random.normal(0, 1, (num_samples, latent_dim))

generated_images = genera.predict(noise)
genera.save("genera_model.h5")

discriminator.save("discriminator_model.h5")

gan.save("gan_model.h5")