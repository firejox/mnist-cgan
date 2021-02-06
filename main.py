import keras
from dataset_generator import MnistDataSet
from gan import GANModel
from gan_monitor import GANMonitor


latent_var_size = 20

dataset = MnistDataSet(latent_var_size=latent_var_size, batch_size=100)
model = GANModel(latent_var_size=latent_var_size)
monitor = GANMonitor(latent_var_size=latent_var_size)
model.compile(
    optimizer={
        'g': keras.optimizers.SGD(),
        'd': keras.optimizers.SGD()
    },
    loss={
        'd': keras.losses.CategoricalCrossentropy(),
        'g': keras.losses.CategoricalCrossentropy()
    }
)

model.fit(dataset, epochs=30, callbacks=[monitor])
