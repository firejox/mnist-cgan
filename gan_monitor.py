from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt


class GANMonitor(Callback):
    def __init__(self, latent_var_size):
        self.label_map = np.fromfunction(
            lambda i, j: np.cos(np.pi * 0.2 * (i * 9.0 + j)) + np.sin(np.pi * 0.2 * (i * 9.0 + j)),
            (10, 10),
            dtype=np.float
        )
        self.latent_var_size = latent_var_size

    def on_epoch_end(self, epoch, logs=None):
        features = np.concatenate([
            self.label_map,
            np.random.normal(size=(10, self.latent_var_size))
        ], axis=1)
        g_images = self.model.image_generator(features)

        fig, ax = plt.subplots(1, 10, figsize=(22, 22))
        for i in range(10):
            image = g_images[i].numpy().reshape((28, 28))
            image = (image * 127.5 + 127.5).astype(np.uint8)

            ax[i].imshow(image, cmap='gray')
            ax[i].set_title('generated image {i}'.format(i=i))
            ax[i].axis('off')

        plt.savefig('mnist-result/image_{i}.png'.format(i=epoch))
        plt.show()
        plt.close()
