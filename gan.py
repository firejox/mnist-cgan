import keras
from keras import Model
import tensorflow as tf
from discriminator import mnist_discriminator
from generator import mnist_generator


class GANModel(Model):
    def __init__(self, latent_var_size):
        super(GANModel, self).__init__()
        self.latent_var_size = latent_var_size
        self.image_generator = mnist_generator(latent_var_size)
        self.discriminator = mnist_discriminator()

    def compile(self, optimizer, loss):
        super(GANModel, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.loss_metrics={
            'd': keras.metrics.Mean(name='d_loss'),
            'g': keras.metrics.Mean(name='g_loss')
        }

    @property
    def metrics(self):
        return [self.loss_metrics['d'], self.loss_metrics['g']]

    def call(self, inputs):
        return inputs

    def train_step(self, data):
        x, y = data
        real_images, features = x
        real_labels, features_labels = y

        with tf.GradientTape(persistent=True) as tape:
            fake_images = self.image_generator(features)

            fake_predicted = self.discriminator(fake_images)
            real_predicted = self.discriminator(real_images)

            d_loss0 = self.loss['d'](features_labels, fake_predicted)
            d_loss1 = self.loss['d'](real_labels, real_predicted)
            d_loss = tf.where(
                d_loss0 < d_loss1,
                -d_loss0 + d_loss1,
                0.1 * d_loss1
            )
            g_loss = self.loss['g'](features_labels, fake_predicted)

        d_grad = tape.gradient(
            d_loss,
            self.discriminator.trainable_variables
        )

        self.optimizer['d'].apply_gradients(
            zip(
                d_grad,
                self.discriminator.trainable_variables
            )
        )

        g_grad = tape.gradient(
            g_loss,
            self.image_generator.trainable_variables
        )

        self.optimizer['g'].apply_gradients(
            zip(
                g_grad,
                self.image_generator.trainable_variables
            )
        )

        self.loss_metrics['d'].update_state(d_loss1)
        self.loss_metrics['g'].update_state(g_loss)

        return {
            'd_loss': self.loss_metrics['d'].result(),
            'g_loss': self.loss_metrics['g'].result()
        }
