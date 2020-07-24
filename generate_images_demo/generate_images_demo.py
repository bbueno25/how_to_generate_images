"""
DOCSTRING
"""
import keras
import matplotlib.pyplot as pyplot
import numpy
import scipy.stats as stats

class GenerateImages:
    """
    DOCSTRING
    """
    def __init__(self):
        self.batch_size = 100
        self.original_dim = 784
        self.latent_dim = 2
        self.intermediate_dim = 256
        self.nb_epoch = 5
        self.epsilon_std = 1.0
        # encoder
        self.x = keras.layers.Input(batch_shape=(self.batch_size, self.original_dim))
        h = keras.layers.Dense(self.intermediate_dim, activation='relu')(self.x)
        self.z_mean = keras.layers.Dense(self.latent_dim)(h)
        self.z_log_var = keras.layers.Dense(self.latent_dim)(h)
        print('Mean:', self.z_mean)
        print('Log Variance:', self.z_log_var)
        z = keras.layers.Lambda(
            self.sampling, output_shape=(self.latent_dim, ))([self.z_mean, self.z_log_var])
        print(z)
        # decoder
        self.decoder_h = keras.layers.Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = keras.layers.Dense(self.original_dim, activation='sigmoid')
        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)
        print(x_decoded_mean)
        self.vae = keras.models.Model(self.x, x_decoded_mean)
        self.vae.compile(optimizer='rmsprop', loss=self.vae_loss)

    def __call__(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), numpy.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), numpy.prod(x_test.shape[1:])))
        self.vae.fit(
            x_train, x_train, shuffle=True, nb_epoch=self.nb_epoch,
            batch_size=self.batch_size, validation_data=(x_test, x_test), verbose=1)
        # build a model to project inputs on the latent space
        encoder = keras.models.Model(self.x, self.z_mean)
        # display a 2D plot of the digit classes in the latent space
        x_test_encoded = encoder.predict(x_test, batch_size=self.batch_size)
        pyplot.figure(figsize=(6, 6))
        pyplot.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test)
        pyplot.colorbar()
        pyplot.show()
        # build a digit generator that can sample from the learned distribution
        decoder_input = keras.layers.Input(shape=(self.latent_dim, ))
        _h_decoded = self.decoder_h(decoder_input)
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        generator = keras.models.Model(decoder_input, _x_decoded_mean)
        # display a 2D manifold of the digits
        n = 15 # figure with 15x15 digits
        digit_size = 28
        figure = numpy.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates on the unit square were transformed 
        # through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, 
        # since the prior of the latent space is Gaussian
        grid_x = stats.norm.ppf(numpy.linspace(0.05, 0.95, n))
        grid_y = stats.norm.ppf(numpy.linspace(0.05, 0.95, n))
        for i, y_index in enumerate(grid_x):
            for j, x_index in enumerate(grid_y):
                z_sample = numpy.array([[x_index, y_index]])
                x_decoded = generator.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
        pyplot.figure(figsize=(10, 10))
        pyplot.imshow(figure, cmap='Greys_r')
        pyplot.show()

    def sampling(self, args):
        """
        DOCSTRING
        """
        self.z_mean, self.z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=(self.batch_size, self.latent_dim), mean=0.0)
        return self.z_mean + keras.backend.exp(self.z_log_var / 2) * epsilon

    def vae_loss(self, x, x_decoded_mean):
        """
        DOCSTRING
        """
        xent_loss = self.original_dim * keras.losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * keras.backend.sum(
            1 + self.z_log_var - keras.backend.square(self.z_mean)
            - keras.backend.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

if __name__ == '__main__':
    generate_images = GenerateImages()
    generate_images()
