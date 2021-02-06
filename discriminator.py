from keras import layers
from keras import Model
import tensorflow as tf
from waverage_pool import WAveragePooling2D


def mnist_discriminator():
    inputs = layers.Input(shape=(28, 28, 1))

    x0 = layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(inputs)
    x0 = layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(x0)

    x1 = WAveragePooling2D(pool_size=(2, 2))(x0)
    x1 = layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(x1)
    x1 = layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(x1)

    x2 = WAveragePooling2D(pool_size=(2, 2))(x1)
    x2 = layers.Conv2D(128, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(x2)
    x2 = layers.Conv2D(128, kernel_size=(2, 2), activation=tf.nn.leaky_relu)(x2)

    x3 = layers.Flatten()(x2)
    outputs = layers.Dense(10, activation='softmax')(x3)

    model = Model(inputs=inputs, outputs=outputs)

    return model
