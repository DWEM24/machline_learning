# 用keras实现FM模型

import keras.backend as K
from keras.layers import Layer, InputSpec


class FMLayer(Layer):
    def __init__(self, factor_order, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)
        self.factor_order = factor_order
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.w = self.add_weight(name='one', shape=(input_dim, 1), initializer='glorot_uniform', trainable=True)
        self.v = self.add_weight(name='two', shape=(input_dim, self.factor_order), initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='bias', shape=(1,), initializer='zeros', trainable=True)
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        X_square = K.square(inputs)
        xv = K.square(K.dot(inputs, self.v))
        xw = K.dot(inputs, self.w)
        p = 0.5 * K.sum(xv - K.dot(X_square, K.square(self.v)), 1)
        rp = K.repeat_elements(K.reshape(p, (-1, 1)), 1, axis=-1)
        output = xw + rp + self.b
        output = K.sigmoid(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], 1
