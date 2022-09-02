import tensorflow as tf


class AddNorm(tf.keras.layers.Layer):
    """Add a residual connection followed by a layer normalization"""

    def __init__(self, norm_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def __call__(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)
