import requests
import gzip
import numpy as np
from keras.utils import Sequence


class MnistDataSet(Sequence):
    def __init__(self, latent_var_size, batch_size):
        def real_mnist_dataset():
            filename = 'mnist.npy'
            try:
                with open(filename, 'rb') as f:
                    x = np.load(f)
                    y = np.load(f)
            except FileNotFoundError:
                x_train_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
                y_train_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
                x_test_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
                y_test_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

                x_train_data = gzip.decompress(requests.get(x_train_url).content)
                y_train_data = gzip.decompress(requests.get(y_train_url).content)
                x_test_data = gzip.decompress(requests.get(x_test_url).content)
                y_test_data = gzip.decompress(requests.get(y_test_url).content)

                train_size = int.from_bytes(x_train_data[4:8], byteorder='big')
                test_size = int.from_bytes(x_test_data[4:8], byteorder='big')
                row = int.from_bytes(x_train_data[8:12], byteorder='big')
                col = int.from_bytes(x_train_data[12:16], byteorder='big')

                x_train = np.frombuffer(
                    x_train_data[16:],
                    dtype=np.uint8
                ).reshape((train_size, row, col, 1))
                y_train = np.frombuffer(y_train_data[8:], dtype=np.uint8)

                x_test = np.frombuffer(
                    x_test_data[16:],
                    dtype=np.uint8
                ).reshape((test_size, row, col, 1))
                y_test = np.frombuffer(y_test_data[8:], dtype=np.uint8)

                x = (np.concatenate([x_train, x_test], axis=0) - 127.5) / 127.5
                y = np.concatenate([y_train, y_test], axis=0)

                with open(filename, 'wb') as f:
                    np.save(f, x)
                    np.save(f, y)

            return x, y

        self.label_map = np.fromfunction(
            lambda i, j: np.cos(np.pi * 0.2 * (i * 9.0 + j)) + np.sin(np.pi * 0.2 * (i * 9.0 + j)),
            (10, 10),
            dtype=np.float
        )

        self.onehot_label_map = np.identity(10, dtype=np.float)

        self.batch_size = batch_size

        x, y = real_mnist_dataset()

        self.data_size = x.shape[0]
        self.latent_var_size = latent_var_size

        self.real_images = x
        self.real_labels = y
        self.real_encoded_labels = self.onehot_label_map[self.real_labels]

        index_labels = np.random.randint(10, size=self.data_size)
        self.features_encoded_labels = self.onehot_label_map[index_labels]
        self.features = np.concatenate([
            self.label_map[index_labels],
            np.random.normal(size=(self.data_size, latent_var_size))
        ], axis=1)

    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, idx):
        batch_real_images = self.real_images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_real_encoded_labels = self.real_encoded_labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_features = self.features[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_features_encoded_labels = self.features_encoded_labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return (batch_real_images, batch_features),\
               (batch_real_encoded_labels, batch_features_encoded_labels)

    def on_epoch_end(self):
        index_labels = np.random.randint(10, size=self.data_size)
        self.features_encoded_labels = self.onehot_label_map[index_labels]
        self.features = np.concatenate([
            self.label_map[index_labels],
            np.random.normal(size=(self.data_size, self.latent_var_size))
        ], axis=1)
