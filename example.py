import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DCGAN import DCGAN
from utils import show_images

# Allow mixed_precision compute if GPU >= RTX 20 series
if False:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

# Load data
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
# Normalize from -1 to 1
train_images = train_images.astype('float32') / 127.5 - 1.
train_dataset = tf.data.Dataset.from_tensor_slices(
                    train_images
                ).shuffle(60000
                ).map(
                    lambda x: tf.image.random_crop(tf.pad((tf.image.resize(tf.expand_dims(x, -1), (32,32))), [[1,1],[1,1],[0,0]], mode='SYMMETRIC'), [32,32,1])
                ).batch(256
                ).prefetch(tf.data.experimental.AUTOTUNE)

noiseDim = 32
N = 180 # Number of images to generate

gan = DCGAN(noiseDim, (32, 32, 1), 1/4, lr=0.0002)
gan.summary()

# Train DC GAN
gan.train(train_dataset, 300, initial_epoch=0, save_gen_freq=1, save_examples_dir='mnist/cutG/', checkpoint_freq=2, checkpoint_dir='mnist/saves/')
# Plot Generator and Discriminator loss history
gan.plotLossHistory()

# Generate images
imgs = gan.generator(
    tf.random.normal((N, noiseDim))
).numpy()
imgs = imgs * 127.5 + 127.5
show_images(imgs)

# Measure FID
print(f"FID: {gan.FID(train_dataset)}")

