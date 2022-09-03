import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P

        self.P = np.zeros((1, max_len, num_hiddens))
        print("P.shape", self.P.shape)
        X = np.arange(max_len, dtype=np.float32).reshape(-1, 1) / np.power(
                10000, np.arange(0, num_hiddens, 2, dtype=np.float32) / num_hiddens
        )

        self.P[:, :, 0::2] = tf.sin(
                X
        )  # x[low::stride] -> positions: 0, 2, 4, ... of all rows and columns
        self.P[:, :, 1::2] = tf.cos(
                X
        )  # x[low::stride] -> positions: 1, 3, 5 , ... of all rows and columns

    def call(self, X, **kwargs):
        # print("X.shape[1]: ", X.shape[1])
        # print("self.P[:, : X.shape[1], :]: ", self.P[:, : X.shape[1], :].shape)
        X = X + self.P[:, : X.shape[1], :]
        return self.dropout(X, **kwargs)
