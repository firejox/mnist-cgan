from keras import Model
from keras import layers
from keras.initializers import Orthogonal
from keras.constraints import UnitNorm


def mnist_generator(latent_var_size):
    inputs = layers.Input(shape=(latent_var_size + 10, ))
    x = layers.Reshape(target_shape=(1, 1, -1))(inputs)

    x = layers.Conv2DTranspose(
        5,
        (2 * 28, 2 * 28),
        kernel_initializer=Orthogonal(),
        kernel_constraint=UnitNorm(axis=[0, 1, 2])
    )(x)

    x = layers.Activation(activation='selu')(x)

    for i in range(14):
        x = layers.Conv2D(
            20,
            (3, 3),
            kernel_initializer=Orthogonal(),
            kernel_constraint=UnitNorm(axis=[0, 1, 2])
        )(x)

        x = layers.Activation(activation='selu')(x)

    x = layers.Conv2D(
        1,
        (1, 1),
        kernel_initializer=Orthogonal(),
        kernel_constraint=UnitNorm(axis=[0, 1, 2])
    )(x)

    outputs = layers.Activation(activation='softsign')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
