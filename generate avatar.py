import tensorflow as tf
import numpy as np
import os

generator = tf.keras.models.load_model('generator_model.h5')  

num_samples = 10  
latent_dim = 100  

for i in range(num_samples):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise)[0]  
    generated_image = (generated_image + 1) / 2.0  

    save_path = f'generated_avatar_{i}.png'
    tf.keras.preprocessing.image.save_img(save_path, generated_image)

print("Avatars generated and saved successfully!")