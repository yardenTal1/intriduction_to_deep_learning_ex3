import time

from matplotlib import pyplot as plt
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses

import numpy as np
import os

latent_dim = 64
noise_sigma = 0.35
train_AE = False
sml_train_size = 50
EPOCHS = 20
BATCH_SIZE = 32
noise_dim = 32

# load train and test images, and pad & reshape them to (-1,32,32,1)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)).astype('float32') / 255.0
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)).astype('float32') / 255.0
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))
print(x_train.shape)
print(x_test.shape)
y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

encoder = Sequential()
encoder.add(layers.Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 1)))
encoder.add(layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Reshape((2 * 2 * 96,)))
encoder.add(layers.Dense(latent_dim))

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
decoder = Sequential()
decoder.add(layers.Dense(2 * 2 * 96, activation='relu', input_shape=(latent_dim,)))
decoder.add(layers.Reshape((2, 2, 96)))
decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='sigmoid', padding='same'))

autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

checkpoint_path = "model_save/cp.ckpt"

if train_AE:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True)
    print("AOUTOENCODER MODEL")
    autoencoder.fit(x_train + noise_sigma * np.random.randn(*x_train.shape), x_train,
                    epochs=15,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[cp_callback])
else:
    autoencoder.load_weights(checkpoint_path)

decoded_imgs = autoencoder.predict(x_test)
latent_codes = encoder.predict(x_test)
decoded_imgs = decoder.predict(latent_codes)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#
# # Your code starts here:
#
# # Classifer Network - currently minimal
# classifier = Sequential()
# classifier.add(layers.Dense(32, activation='softmax', input_shape=(latent_dim,)))
# classifier.add(layers.Dense(10, activation='softmax', input_shape=(32,)))
# # classifier.add(layers.Dense(10,activation='softmax', input_shape=(16,)))
#
# train_codes = encoder.predict(x_train[:sml_train_size])
# test_codes = encoder.predict(x_test)
#
# classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# print("TRANSFER LEARNING - ENCODER + CLASSIFIER MODEL")
# classifier.fit(train_codes, y_train[:sml_train_size],
#                epochs=200,
#                batch_size=16,
#                shuffle=True,
#                validation_data=(test_codes, y_test))
#
# full_cls_enc = keras.models.clone_model(encoder)
# full_cls_cls = keras.models.clone_model(classifier)
# full_cls = keras.Model(full_cls_enc.inputs, full_cls_cls(full_cls_enc.outputs))
#
# full_cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print("CLASSIFIER MODEL")
# full_cls.fit(x_train[:sml_train_size], y_train[:sml_train_size],
#              epochs=100,
#              batch_size=16,
#              shuffle=True,
#              validation_data=(x_test, y_test))


def get_generator_model():
    # GAN - q3
    MLP_generator = Sequential()
    MLP_generator.add(layers.Dense(64, activation='relu', input_shape=(noise_dim,)))
    MLP_generator.add(layers.Dense(128, activation='relu'))
    MLP_generator.add(layers.Dense(128, activation='relu'))
    MLP_generator.add(layers.Dense(latent_dim))

    return MLP_generator


def get_discriminator_model():
    # Discriminator
    discriminator = Sequential()
    discriminator.add(layers.Dense(32, activation='relu', input_shape=(latent_dim,)))
    discriminator.add(layers.Dense(16, activation="relu"))
    discriminator.add(layers.Dense(1))

    return discriminator

def get_conditional_generator_model():
    # GAN - q3
    MLP_generator = Sequential()
    MLP_generator.add(layers.Dense(64, activation='relu', input_shape=(noise_dim + 10,)))
    MLP_generator.add(layers.Dense(128, activation='relu'))
    MLP_generator.add(layers.Dense(128, activation='relu'))
    MLP_generator.add(layers.Dense(latent_dim))

    return MLP_generator


def get_conditional_discriminator_model():
    # Discriminator
    discriminator = Sequential()
    discriminator.add(layers.Dense(32, activation='relu', input_shape=(latent_dim + 10,)))
    discriminator.add(layers.Dense(16, activation="relu"))
    discriminator.add(layers.Dense(1))

    return discriminator


def get_discriminator_loss(real_output_dis, fake_output_dis):
    real_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output_dis), real_output_dis)
    fake_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output_dis), fake_output_dis)
    return real_loss + fake_loss


def get_generator_loss(fake_output_dis):
    return keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output_dis), fake_output_dis)


def preprocess_data(x_train, y_train):
    batched_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
    return batched_data



def train_gan(train_set):
    discriminator_accuracy = tf.keras.metrics.BinaryAccuracy()
    generator_success_accuracy = tf.keras.metrics.BinaryAccuracy()
    generator_loss = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
    generator_optimizer = keras.optimizers.Adam()
    discriminator_optimizer = keras.optimizers.Adam()

    generator = get_generator_model()
    discriminator = get_discriminator_model()

    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch, label_batch in train_set:
            # noise = tf.random.normal([BATCH_SIZE, noise_dim], stddev=noise_sigma)
            noise = tf.random.normal([BATCH_SIZE, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_latent_vector = generator(noise, training=True)

                real_latent_vector = encoder.predict(image_batch)

                real_output = discriminator(real_latent_vector, training=True)
                fake_output = discriminator(generated_latent_vector, training=True)

                real_output_sig = tf.sigmoid(real_output)
                fake_output_sig = tf.sigmoid(fake_output)

                gen_loss = get_generator_loss(fake_output)
                disc_loss = get_discriminator_loss(real_output, fake_output)

                discriminator_accuracy.update_state(tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], 0),
                                                    tf.concat([real_output_sig, fake_output_sig], 0))

                generator_success_accuracy.update_state(tf.ones_like(fake_output), fake_output_sig)
                generator_loss.update_state(tf.ones_like(fake_output), fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


        # Produce images for the GIF as we go
        # display.clear_output(wait=True)
        # generate_and_save_images(generator,epoch + 1,seed)

        # Save the model every 15 epochs
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)
        fake_images = decoder.predict(generated_latent_vector)
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(1, n + 1):
            # Display original
            ax = plt.subplot(2, n, i)
            plt.imshow(fake_images[i].reshape(32, 32))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('Total running discriminator accuracy so far: %.3f' % discriminator_accuracy.result())
        print('Total running generator accuracy so far: %.3f' % generator_success_accuracy.result())
        print('Total running generator loss so far: %.3f' % generator_loss.result())

        discriminator_accuracy.reset_states()
        generator_success_accuracy.reset_states()
        generator_loss.reset_states()

    generator.save(r'model_save\generator')
    discriminator.save(r'model_save\discriminator')


def train_conditional_gan(train_set):
    discriminator_accuracy = tf.keras.metrics.BinaryAccuracy()
    generator_success_accuracy = tf.keras.metrics.BinaryAccuracy()
    generator_loss = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
    generator_optimizer = keras.optimizers.Adam()
    discriminator_optimizer = keras.optimizers.Adam()

    generator = get_conditional_generator_model()
    discriminator = get_conditional_discriminator_model()

    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch, label_batch in train_set:
            # noise = tf.random.normal([BATCH_SIZE, noise_dim], stddev=noise_sigma)
            noise = tf.random.normal([BATCH_SIZE, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                indices = np.random.random_integers(0, 9, BATCH_SIZE)
                fake_one_hot_vector = tf.one_hot(indices, 10)
                generated_latent_vector = generator(tf.concat([noise, fake_one_hot_vector], axis=1), training=True)

                real_latent_vector = encoder.predict(image_batch)
                real_one_hot_vector = label_batch

                real_output = discriminator(tf.concat([real_latent_vector, real_one_hot_vector], axis=1), training=True)
                fake_output = discriminator(tf.concat([generated_latent_vector, fake_one_hot_vector], axis=1), training=True)

                real_output_sig = tf.sigmoid(real_output)
                fake_output_sig = tf.sigmoid(fake_output)

                gen_loss = get_generator_loss(fake_output)
                disc_loss = get_discriminator_loss(real_output, fake_output)

                discriminator_accuracy.update_state(tf.concat([tf.ones_like(real_output), tf.zeros_like(fake_output)], 0),
                                                    tf.concat([real_output_sig, fake_output_sig], 0))

                generator_success_accuracy.update_state(tf.ones_like(fake_output), fake_output_sig)
                generator_loss.update_state(tf.ones_like(fake_output), fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        noise_to_show = tf.random.normal([10, noise_dim])
        fake_one_hot_vector_to_show = tf.one_hot(list(range(0, 10)), 10)
        generated_latent_vector_to_show = generator(tf.concat([noise_to_show, fake_one_hot_vector_to_show], axis=1), training=True)


        fake_images = decoder.predict(generated_latent_vector_to_show)
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(1, n + 1):
            # Display original
            ax = plt.subplot(2, n, i)
            plt.imshow(fake_images[i-1].reshape(32, 32))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('Total running discriminator accuracy so far: %.3f' % discriminator_accuracy.result())
        print('Total running generator accuracy so far: %.3f' % generator_success_accuracy.result())
        print('Total running generator loss so far: %.3f' % generator_loss.result())

        discriminator_accuracy.reset_states()
        generator_success_accuracy.reset_states()
        generator_loss.reset_states()

    generator.save(r'model_save\cond_generator')
    discriminator.save(r'model_save\cond_discriminator')


def interpolation_comparison(encoder, decoder):
    generator = keras.models.load_model(r'model_save\generator')
    noise = tf.random.normal([2, noise_dim])
    output_generator = generator.predict(noise)
    output_encoder = encoder(x_test[:2])

    L1_gan = output_generator[0]
    L2_gan = output_generator[1]
    L1_encoder = output_encoder[0]
    L1_encoder = output_encoder[1]

    interpolation_arr = []
    for a in range(0,11):
        interpolation_arr.append(L1_gan*a/10 + (1-a/10)*L2_gan)
    decoder_interpolation = decoder.predict(np.array(interpolation_arr))

    n = 11
    plt.figure(figsize=(20, 4))
    for i in range(0, n):
        # Display original
        ax = plt.subplot(1, n, i+1)
        plt.imshow(decoder_interpolation[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    #     # Generate after the final epoch
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                          epochs,
    #                          seed)



batched_data = preprocess_data(x_train, y_train)
# train_gan(batched_data)
#
# interpolation_comparison(encoder, decoder)

train_conditional_gan(batched_data)
