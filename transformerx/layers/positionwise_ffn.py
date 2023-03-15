import os

import tensorflow as tf


# class PositionWiseFFN(tf.keras.layers.Layer):
#     """Compute position-wise feed-forward network [1]_.
#
#     Consists of two dense layers with ReLU activation.
#
#     See Also
#     --------
#     transformerx.layers.transformer_encoder_block
#     transformerx.layers.transformer_decoder_block
#
#     Notes
#     -----
#     Position-Wise Feed-Forward Layer is a type of feedforward layer consisting of two dense layers that applies to the
#     last dimension, which means the same dense layers are used for each position item in the sequence, so called
#     position-wise.
#
#     Parameters
#     ----------
#     input_hidden_units :
#         Number of input hidden units
#     output_hidden_units :
#         Number of output hidden units
#
#     Returns
#     -------
#     Output :
#         A tensor of shape (batch size, number of time steps or sequence length in tokens, number of hidden units or
#         feature dimension)
#
#     Examples
#     --------
#     >>> tf.random.set_seed(1)
#     >>> ffn = PositionWiseFFN(6, 4)
#     >>> x = tf.ones((2, 3, 6))
#     >>> print(ffn(x))
#     tf.Tensor(
#     [[[ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]
#       [ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]
#       [ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]]
#      [[ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]
#       [ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]
#       [ 0.51875997 -0.2624486  -0.79755557  1.5191057 ]]], shape=(2, 3, 4), dtype=float32)
#
#     References
#     ----------
#     .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I.
#     (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
#     """
#
#     def __init__(self, input_hidden_units, output_hidden_units):
#         super().__init__()
#         self.dense1 = tf.keras.layers.Dense(input_hidden_units)
#         self.relu = tf.keras.layers.ReLU()
#         self.dense2 = tf.keras.layers.Dense(output_hidden_units)
#
#     def call(self, x):
#         # x.shape: (batch size, number of time steps or sequence length in tokens, number of hidden units or
#         # feature dimension)
#         return self.dense2(self.relu(self.dense1(x)))
#
#



class PositionWiseFFN(tf.keras.layers.Layer):

    def __init__(self, input_hidden_units, output_hidden_units, activation='relu', init='glorot_uniform',
                 non_linear_proj=None, contextualized_embeddings=None):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(input_hidden_units, activation=activation, kernel_initializer=init)
        self.non_linear_proj = non_linear_proj
        if self.non_linear_proj == 'glu':
            self.glu = tf.keras.layers.Dense(output_hidden_units * 2, activation='sigmoid', kernel_initializer=init)
        elif self.non_linear_proj == 'selu':
            self.selu = tf.keras.layers.Dense(output_hidden_units * 2, activation='selu', kernel_initializer=init)
        else:
            self.dense2 = tf.keras.layers.Dense(output_hidden_units, activation=activation, kernel_initializer=init)
        self.contextualized_embeddings = contextualized_embeddings

    def call(self, x):
        # x.shape: (batch size, number of time steps or sequence length in tokens, number of hidden units or
        # feature dimension)
        if self.contextualized_embeddings is not None:
            bert_output = self.contextualized_embeddings(x)
            x = bert_output[0]
        if self.non_linear_proj == 'glu':
            x = self.dense1(x)
            split_units = x.shape[-1] // 2
            return x[:, :, :split_units] * tf.keras.activations.sigmoid(self.glu(x[:, :, split_units:]))[:, :,
                                           :split_units]
        elif self.non_linear_proj == 'selu':
            x = self.dense1(x)
            split_units = x.shape[-1] // 2
            return x[:, :, :split_units] * tf.keras.activations.sigmoid(self.selu(x[:, :, split_units:]))[:, :,
                                           :split_units]
        else:
            return self.dense2(self.dense1(x))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == '__main__':
    tf.random.set_seed(1)
    # ffn = PositionWiseFFN(6, 4)
    ffn = PositionWiseFFN(input_hidden_units=32, output_hidden_units=64, activation='tanh', init='he_uniform',
                          non_linear_proj='glu')
    x = tf.ones((2, 3, 32))
    print(ffn(x))

