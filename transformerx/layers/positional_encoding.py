import numpy as np
import tensorflow as tf


class AbsolutePositionalEncoding(tf.keras.layers.Layer):
    """Compute absolute positional encoding object [1]_.

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
        Embedding Size; Length of the positional encoding's hidden units, the same as the length of Embedding.
    dropout_rate :
        Float between 0 and 1. Fraction of the input units to drop.
    max_len :
        Maximum length of the steps to calculate sinusoid

    Returns
    -------
    output:
        Tensor of the same shape of the input tensor with positinoal encodings added

    Examples
    --------
    >>> depth, num_steps = 32, 50
    >>> pos_encoding = AbsolutePositionalEncoding(depth, dropout_rate=0.2)
    >>> x = tf.zeros((1, num_steps, depth))
    >>> print(x.shape)
    (1, 50, 32)
    >>> x = pos_encoding(x, training=False)
    >>> P = pos_encoding.P[:, : x.shape[1], :]
    >>> print(x.shape)
    (1, 50, 32)
    >>> print(P.shape)
    (1, 50, 32)

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I.
    (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
    """

    def __init__(self, depth, dropout_rate=0, max_len=1000):
        super(AbsolutePositionalEncoding, self).__init__()
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

def exists(val):
    return val is not None

class FixedPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(FixedPositionalEncoding, self).__init__()
        inv_freq = 1. / (10000 ** (np.arange(0, dim, 2).__float__() / dim))

    def call(self, inputs, pos=None, seq_dim=1, offset=0, **kwargs):
        if not exists(pos):
            pos = np.arange(input.shape[seq_dim])
