import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, InputLayer,Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.utils import to_categorical


def binomial_loss(x, x_decoded_mean, t_mean, t_log_var, batch_size):
    KL = (-t_log_var + tf.exp(t_log_var) + t_mean**2 - 1.)/2.
    KL = tf.reduce_sum(KL, axis=-1)
    rec_loss = x * tf.math.log(x_decoded_mean + 10**(-19)) + \
        (1 - x) * tf.math.log(1 - x_decoded_mean + 10**(-19))
    rec_loss = tf.reduce_sum(tf.reshape(rec_loss, (batch_size, -1)), axis=-1)
    return tf.reduce_mean(KL - rec_loss)


def create_encoder(input_dim):
    encoder = Sequential(name='encoder')
    encoder.add(InputLayer(input_dim))
    encoder.add(Conv2D(32, (3, 3), activation="relu"))
    encoder.add(Conv2D(32, (3, 3), activation="relu", name="conv"))
    encoder.add(Flatten())
    encoder.add(Dense(intermediate_dim, activation='relu'))
    encoder.add(Dense(2 * latent_dim))
    return encoder


def sampling(args):
    t_mean, t_log_var = args
    ksi = tf.random.normal(t_mean.shape)
    return t_mean + ksi*tf.exp(0.5*t_log_var)


def create_decoder(input_dim, conv_shape):
    decoder = Sequential(name='decoder')
    decoder.add(InputLayer((input_dim,)))
    decoder.add(Dense(intermediate_dim, activation='relu'))
    decoder.add(Dense(np.prod(conv_shape[1:]), activation='relu'))
    decoder.add(Reshape(conv_shape[1:]))
    decoder.add(Conv2DTranspose(32, (3, 3), activation="relu"))
    decoder.add(Conv2DTranspose(1, (3, 3), activation="sigmoid"))
    return decoder


if __name__ == "__main__":
    batch_size = 100
    original_dim = (28, 28, 1)
    latent_dim = 15
    intermediate_dim = 256
    epochs = 10
    
    x = Input(batch_shape=(batch_size, 28, 28, 1))
    
    encoder = create_encoder(original_dim)
    
    get_t_mean = Lambda(lambda h: h[:, :latent_dim])
    get_t_log_var = Lambda(lambda h: h[:, latent_dim:])
    h = encoder(x)
    t_mean = get_t_mean(h)
    t_log_var = get_t_log_var(h)
    
    t = Lambda(sampling)([t_mean, t_log_var])
    
    conv_shape = encoder.get_layer(name="conv").get_output_shape_at(0)
    
    decoder = create_decoder(latent_dim, conv_shape)
    x_decoded_mean = decoder(t)
    
    loss = binomial_loss(x, x_decoded_mean, t_mean, t_log_var, batch_size)
    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer=keras.optimizers.Adam(), loss=lambda x, y: loss)
    
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))
    
    hist = vae.fit(x=x_train, y=x_train,
                   shuffle=True,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(x_test, x_test),
                   verbose=2)

    fig = plt.figure(figsize=(10, 10))
    for fid_idx, (data, title) in enumerate(
                zip([x_train, x_test], ['Train', 'Validation'])):
        n = 10
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * 2))
        decoded = vae.predict(data[:batch_size, :], batch_size=batch_size)
        for i in range(10):
            figure[i * digit_size: (i + 1) * digit_size,
                   :digit_size] = data[i, :].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   digit_size:] = decoded[i, :].reshape(digit_size, digit_size)
        ax = fig.add_subplot(1, 2, fid_idx + 1)
        ax.imshow(figure, cmap='Greys_r')
        ax.set_title(title)
        ax.axis('off')
    plt.show()