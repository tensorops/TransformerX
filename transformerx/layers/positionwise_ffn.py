import os

import tensorflow as tf


class PositionWiseFFN(tf.keras.layers.Layer):
    """Compute position-wise feed-forward network [1]_.

    Consists of two dense layers with ReLU activation.

    See Also
    --------
    transformerx.layers.transformer_encoder_block
    transformerx.layers.transformer_decoder_block

    Notes
    -----
    Position-Wise Feed-Forward Layer is a type of feedforward layer consisting of two dense layers that applies to the
    last dimension, which means the same dense layers are used for each position item in the sequence, so called
    position-wise.

    Parameters
    ----------
    input_hidden_units :
        Number of input hidden units
    output_hidden_units :
        Number of output hidden units

    Returns
    -------
    Output :
        A tensor of shape (batch size, number of time steps or sequence length in tokens, number of hidden units or
        feature dimension)

    Examples
    --------
    >>> tf.random.set_seed(1)
    >>> ffn = PositionWiseFFN(6, 4)
    >>> x = tf.ones((2, 3, 6))
    >>> print(ffn(x))
    tf.Tensor(
    [[[ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]
      [ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]
      [ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]]
     [[ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]
      [ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]
      [ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]]], shape=(2, 3, 4), dtype=float32)

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I.
    (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
    """

    def __init__(self, input_hidden_units, output_hidden_units):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(input_hidden_units)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(output_hidden_units)

    def call(self, x):
        # x.shape: (batch size, number of time steps or sequence length in tokens, number of hidden units or
        # feature dimension)
        return self.dense2(self.relu(self.dense1(x)))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == '__main__':
    tf.random.set_seed(1)
    ffn = PositionWiseFFN(6, 4)
    x = tf.ones((2, 3, 6))
    print(ffn(x))
