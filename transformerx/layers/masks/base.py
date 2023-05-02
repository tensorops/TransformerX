import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, inputs, *args, **kwargs):
        mask = self.build_mask(inputs.shape)
        return inputs * mask
