import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, inputs, *args, **kwargs):
        mask = self.build_mask(inputs.shape)
        return inputs * mask


class AttentionMask(BaseMask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        seq_len = input_shape[1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return tf.expand_dims(mask, axis=0)
