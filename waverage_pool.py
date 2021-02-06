import tensorflow as tf
from keras import backend
from keras.utils import conv_utils
from keras import activations
from keras.engine.base_layer import Layer
from keras.layers import InputSpec
from tensorflow.python.framework import tensor_shape


class WAveragePooling2D(Layer):
    def __init__(self, weight_function='softplus', pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        super(WAveragePooling2D, self).__init__(**kwargs)

        def normalize_data_format(value):
            if value is None:
                value = backend.image_data_format()
            _data_format = value.lower()
            if _data_format not in {'channels_first', 'channels_last'}:
                raise ValueError('The `data_format` argument must be one of '
                                 '"channels_first", "channels_last". Received: ' +
                                 str(value))
            return _data_format

        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size

        self.weight_function = activations.get(weight_function)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        def convert_data_format(data_format, ndim):
            if data_format == 'channels_last':
                if ndim == 3:
                    return 'NWC'
                elif ndim == 4:
                    return 'NHWC'
                elif ndim == 5:
                    return 'NDHWC'
                else:
                    raise ValueError('Input rank not supported:', ndim)
            elif data_format == 'channels_first':
                if ndim == 3:
                    return 'NCW'
                elif ndim == 4:
                    return 'NCHW'
                elif ndim == 5:
                    return 'NCDHW'
                else:
                    raise ValueError('Input rank not supported:', ndim)
            else:
                raise ValueError('Invalid data_format:', data_format)

        if self.data_format == 'channels_last':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides

        weights = self.weight_function(inputs)
        w_inputs = tf.math.multiply_no_nan(weights, inputs)

        avg_w = tf.nn.avg_pool(
            weights,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=convert_data_format(self.data_format, 4)
        )

        avg_w_inputs = tf.nn.avg_pool(
            w_inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=convert_data_format(self.data_format, 4)
        )

        outputs = tf.math.divide_no_nan(avg_w_inputs, avg_w)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        else:
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])

    def get_config(self):
        config = {
            'weight_function': self.weight_function,
            'pool_size': self.pool_size,
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format
        }

        base_config = super(WAveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
