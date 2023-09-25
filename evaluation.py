import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import os

incept_mod = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
incept_mod = tf.keras.Model(inputs=incept_mod.input, outputs=incept_mod.layers[-1].output)

def preprocess_image(image):
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

reference_data = np.load('reference_dataset.npy') 

def calculate_inception_score(images, model, batch_size=32):
    images = np.array([preprocess_image(img) for img in images])
    activations = model.predict(images)

    p_yx = np.exp(activations) / np.exp(activations).sum(axis=1, keepdims=True)
    KL_divs = p_yx * (np.log(p_yx) - np.log(p_yx.mean(axis=0, keepdims=True)))
    KL_divergence = KL_divs.sum(axis=1)
    inception_score = np.exp(KL_divergence.mean())

    return inception_score

inception_scores = calculate_inception_score(generated_images, incept_mod)
print(f"Inception Scores: {inception_scores}")

def calculate_fid(real_data, generated_data, model, batch_size=32):
    real_activations = model.predict(real_data)
    generated_activations = model.predict(generated_data)

    m1 = real_activations.mean(axis=0)
    m2 = generated_activations.mean(axis=0)

    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(generated_activations, rowvar=False)

    diff = m1 - m2
    covmean, _ = tfa.metrics.FID._sqrtm(s1.dot(s2), eps=1e-6)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = np.sum(diff**2) + np.trace(s1 + s2 - 2*covmean)

    return fid_score

fid_scores = calculate_fid(reference_data, generated_images, incept_mod)
print(f"FID Scores: {fid_scores}")