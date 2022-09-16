import numpy as np
import tensorflow as tf


class AbsolutePositionalEncoding(tf.keras.layers.Layer):
    """Absolute positional encoding object.

    Generate a sinusoid for each dimension of the positional encoding where wavelengths form a geometric progression
    from :math:`2π` to :math:`(10000)2π`.

    Notes
    -----
    Absolute Position Encodings are a type of position embeddings for [Transformer-based models] where positional
    encodings are added to the input embeddings at the bottoms of the encoder and decoder stacks. The positional
    encodings have the same dimension :math:`d_model` as the embeddings, so that the two can be summed. In the original
    implementation, sine and cosine functions of different frequencies are used:

    .. math::
        PE(pos, 2i) = \sin(pos^{2i/d_{model}})

        PE(pos, 2i+1) = \cos(pos^{2i/d_{model}})

    where  is the position and  is the dimension.
    
    
    Parameters
    ----------
    depth :
        Length of the positional encoding's hidden units.
    dropout_rate :
        Float between 0 and 1. Fraction of the input units to drop.
    max_len :
        Maximum length of the steps to calculate sinusoid

    Examples
    --------
    >>> depth, num_steps = 32, 50
    >>> pos_encoding = AbsolutePositionalEncoding(depth, dropout_rate=0.2)
    >>> x = tf.zeros((1, num_steps, depth))
    >>> print(x.shape)
    (1, 50, 32)
    >>> X = pos_encoding(x, training=False)
    >>> P = pos_encoding.P[:, : X.shape[1], :]
    >>> print(X.shape)
    (1, 50, 32)
    >>> print(P.shape)
    (1, 50, 32)
    """

    def __init__(self, depth, dropout_rate=0, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # Create a long enough P

        self.P = np.zeros((1, max_len, depth))
        X = np.arange(max_len, dtype=np.float32).reshape(-1, 1) / np.power(
            10000, np.arange(0, depth, 2, dtype=np.float32) / depth
        )

        self.P[:, :, 0::2] = tf.sin(
            X
        )  # x[low::stride] -> positions: 0, 2, 4, ... of all rows and columns
        self.P[:, :, 1::2] = tf.cos(
            X
        )  # x[low::stride] -> positions: 1, 3, 5 , ... of all rows and columns

    def call(self, X, **kwargs):
        # print("x.shape[1]: ", x.shape[1])
        # print("self.P[:, : x.shape[1], :]: ", self.P[:, : x.shape[1], :].shape)
        X = X + self.P[:, : X.shape[1], :]
        return self.dropout(X, **kwargs)
