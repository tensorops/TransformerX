import tensorflow as tf


class PositionWiseFFN(tf.keras.layers.Layer):
    """Position-wise feed-forward network."""

    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        # x.shape: (batch size, number of time steps or sequence length in tokens, number of hidden units or feature dimension)
        return self.dense2(self.relu(self.dense1(X)))
