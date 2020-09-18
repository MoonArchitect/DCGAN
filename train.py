import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Enable XLA
#os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DCGAN import DCGAN

# Allow mixed_preecision compute
if False:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)


gan = DCGAN(
    noise_dim=128,
    image_shape=(128, 128, 3),
    width_multiplier=1
)

gan.summary()
#gan.load(saved_generator_path='saves/gen/', saved_discriminator_path='saves/disc/')

# Load dataset
#train_images = np.load("faces.npy")
#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).cache().shuffle(60000).batch(256).map(lambda x: (tf.cast(x, tf.float32) - 127.5) / 127.5, num_parallel_calls=12).prefetch(5)

gan.train(
    train_dataset, 
    epochs=1500, initial_epoch=0, 
    save_examples_dir='gen_faces/'
)


