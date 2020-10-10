import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as nn
import matplotlib.pyplot as plt

from scipy.linalg import sqrtm
import os

class DCGAN():
    """
    """
    def __init__(self,
                 noise_dim,
                 image_shape,
                 width_multiplier = 1,
                 lr=0.0002,
                 beta1=0.5):
        
        self.noise_dim = noise_dim
        self.image_shape = image_shape
        self.width_multiplier = width_multiplier

        self.cross_entropy = nn.losses.BinaryCrossentropy(from_logits=True)

        self.generator = self.get_generator(noise_dim, image_shape, width_multiplier)
        self.discriminator = self.get_discriminator(image_shape, width_multiplier)

        self.generator_optimizer = nn.optimizers.Adam(lr, beta_1=beta1)
        self.discriminator_optimizer = nn.optimizers.Adam(lr, beta_1=beta1)

        self.loss_history = []


    def get_generator(self, noise_dim, image_shape, width_multiplier):
        img_H, img_W, img_channels = image_shape

        model = nn.Sequential(name='Generator')
        
        model.add(nn.layers.Dense(int(4*4*1024*width_multiplier), use_bias=False, input_shape=(noise_dim,)))
        model.add(nn.layers.Reshape((4, 4, int(1024*width_multiplier))))

        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))
        model.add(nn.layers.Conv2DTranspose(int(512*width_multiplier), (5, 5), strides=(2, 2), padding='same', use_bias=False))

        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))
        model.add(nn.layers.Conv2DTranspose(int(256*width_multiplier), (5, 5), strides=(2, 2), padding='same', use_bias=False))

        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))
        model.add(nn.layers.Conv2DTranspose(int(128*width_multiplier), (5, 5), strides=(2, 2), padding='same', use_bias=False))
        
        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))
        model.add(nn.layers.Conv2DTranspose(int(64*width_multiplier), (5, 5), strides=(1, 1), padding='same', use_bias=False))

        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))
        model.add(nn.layers.Conv2DTranspose(img_channels, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

        return model

    def get_discriminator(self, image_shape, width_multiplier):
        model = nn.Sequential(name='Discriminator')
        model.add(nn.layers.Conv2D(int(64*width_multiplier), (5, 5), strides=(2, 2), padding='same', input_shape=image_shape))
        model.add(nn.layers.LeakyReLU(0.2))

        model.add(nn.layers.Conv2D(int(128*width_multiplier), (5, 5), strides=(2, 2), padding='same'))
        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))

        model.add(nn.layers.Conv2D(int(256*width_multiplier), (5, 5), strides=(2, 2), padding='same'))
        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))
        model.add(nn.layers.Dropout(0.1))

        model.add(nn.layers.Conv2D(int(512*width_multiplier), (5, 5), strides=(2, 2), padding='same'))
        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))
        model.add(nn.layers.Dropout(0.1))

        model.add(nn.layers.Conv2D(int(1024*width_multiplier), (5, 5), strides=(1, 1), padding='same'))
        model.add(nn.layers.BatchNormalization())
        model.add(nn.layers.LeakyReLU(0.2))
        model.add(nn.layers.Dropout(0.1))

        model.add(nn.layers.Flatten())
        model.add(nn.layers.Dense(1))
        
        return model

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss, fake_loss

    @tf.function
    def train_step(self, data_batch, step, epoch):
        noise = tf.random.normal([data_batch.shape[0], self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            fake_output = self.discriminator(generated_images, training=True)
            real_output = self.discriminator(data_batch, training=True)

            gen_loss = self.generator_loss(fake_output)
            real_disc_loss, fake_disc_loss = self.discriminator_loss(real_output, fake_output)
            disc_loss = real_disc_loss + fake_disc_loss
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, real_disc_loss, fake_disc_loss

    def train(self, 
              dataset, 
              epochs,
              initial_epoch=0,
              save_gen_freq=2,
              save_gen_examples=49,
              save_gen_seed=16424432,
              save_examples_dir='gen_images',
              checkpoint_freq=15, 
              checkpoint_dir='saves/'):
        """
        """
        noise = tf.random.normal([save_gen_examples, self.noise_dim], seed=save_gen_seed)
        for epoch in range(initial_epoch, epochs):
            start = time.time()
            
            avg = np.array([0., 0., 0.])
            step = 0
            for step, image_batch in enumerate(dataset):
                batch_size = image_batch.shape[0]
                stats = self.train_step(image_batch, tf.constant(step), tf.constant(epoch))
                
                avg += [stat.numpy() for stat in stats]
                print("Epoch:{:04d}| Step:{:03d}|  Gen_loss: {:4f}     Disc_R_loss: {:4f}     Disc_F_loss:{:4f}".format(epoch, step, *(avg / (step + 1))), end='\r')
            print("Epoch:{:04d}| Step:{:03d}|  Gen_loss: {:4f}     Disc_R_loss: {:4f}     Disc_F_loss:{:4f}".format(epoch, step, *(avg / (step + 1))), end='     ')
            self.loss_history += [avg / (step + 1)]
            
            if save_gen_freq and epoch % save_gen_freq == 0:
                self.save_generated_images(epoch, noise, save_examples_dir, save_gen_examples)

            print ('{:3f} sec'.format(time.time()-start))
            if checkpoint_freq and (epoch + 1) % checkpoint_freq == 0:
                self.save(epoch=epoch+1, checkpoint_dir=checkpoint_dir)

        print(" -->> Training is finished")

    def save(self, epoch, 
             checkpoint_dir='saves/'):
        """
        """ 
        if checkpoint_dir:
            self.generator.save_weights(os.path.join(checkpoint_dir,f"generator/generator_{epoch}"))
            print(f"Generator is saved to {checkpoint_dir}generator/generator_{epoch}")

            self.discriminator.save_weights(os.path.join(checkpoint_dir,f"discriminator/discriminator_{epoch}"))
            print(f"Discriminator is saved to {checkpoint_dir}discriminator/discriminator_{epoch}")
    def load(self, saved_generator_path=None, saved_discriminator_path=None):
        """
        """
        if saved_generator_path:
            self.generator.load_weights(saved_generator_path)
            print(f"Generator is loaded from {saved_generator_path}")

        if saved_discriminator_path:
            self.discriminator.save_weights(saved_discriminator_path)
            print(f"Discriminator is loaded from {saved_discriminator_path}")
    
    def save_generated_images(self, epoch, test_input, dir, n_images=49):
        predictions = self.generator(test_input, training=False)
        axis_length = (np.sqrt(n_images),)*2

        fig = plt.figure(figsize=axis_length)

        for i in range(predictions.shape[0]):
            plt.subplot(*axis_length, i+1)
            img = (predictions[i, :, :, :] * 127.5 + 127.5).numpy().astype('uint8')
            cmap = None
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = np.squeeze(img, -1)
                cmap = 'gray'
            plt.imshow(img, cmap)
            plt.axis('off')

        
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(os.path.join(dir, f"image_at_epoch_{epoch:04d}.png"))

        plt.close()

    @tf.function
    def FID_step(self, data_batch):
        noise = tf.random.normal((data_batch.shape[0], self.noise_dim))
        gen_data_batch = self.generator(noise)

        data_batch = tf.image.resize(data_batch, (299, 299))
        gen_data_batch = tf.image.resize(gen_data_batch, (299, 299))
        
        if data_batch.shape[-1] == 1: 
            data_batch = tf.repeat(data_batch, [3], -1)
            gen_data_batch = tf.repeat(gen_data_batch, [3], -1)

        real_features = self.inceptionV3(data_batch)
        gen_features = self.inceptionV3(gen_data_batch)
        return real_features, gen_features

    def FID(self, image_dataset, samples = None):
        """
        """
        assert len(image_dataset.element_spec.shape) == 4, "Images(3d tensor) from Dataset have to be in batches"

        deleteInceptionOnExit = False
        if not hasattr(self, 'inceptionV3'):
            self.inceptionV3 = nn.applications.InceptionV3(include_top=False, input_shape=(299,299,3), pooling='avg')
            deleteInceptionOnExit = True
        
        real_features = np.empty((0,2048), float)
        gen_features = np.empty((0,2048), float)
        iter_start = time.time()
        for step, img_batch in enumerate(image_dataset):
            if samples and step >= samples:
                break

            print('\x1b[1K\r', end='') # clear previous line
            step_start = time.time()

            r_features, g_features = self.FID_step(img_batch)
            real_features = np.append(real_features, r_features.numpy(), axis=0)
            gen_features = np.append(gen_features, g_features.numpy(), axis=0)

            if samples:
                print(f"Step:{step:03d}/{samples}  |  {time.time()-step_start:3f} sec/step  |  ETA: {((time.time()-step_start) * (samples - step)):3f} sec. ")
            else:
                print(f"Step:{step:03d}  |  {time.time()-step_start:3f} sec/step ")

        print ('{:3f} sec'.format(time.time()-iter_start))

        if deleteInceptionOnExit:
            del self.inceptionV3
        
        meanR, sigmaR = np.mean(real_features), np.cov(real_features, rowvar=False)
        meanG, sigmaG = np.mean(gen_features), np.cov(gen_features, rowvar=False)

        mean_diff = np.sum((meanR - meanG)**2)
        covmean = sqrtm(np.dot(sigmaR, sigmaG))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return mean_diff + np.trace(sigmaR + sigmaG - 2.0 * covmean)
