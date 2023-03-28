import os

import tensorflow as tf


# todo: remove commented old implementation after writing tests
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


class PositionwiseFFN(tf.keras.layers.Layer):
    """
    Position-wise feed-forward network layer.

    Consists of two dense layers with customizable activation functions, weight initialization, non-linear projection,
    and contextualized embeddings.

    Parameters
    ----------
    input_hidden_units : int
        Number of input hidden units.
    output_hidden_units : int
        Number of output hidden units.
    activation : str or callable, optional
        Activation function to use in the dense layers. Default is 'relu'.
    kernel_initializer : str or callable, optional
        Weight initialization strategy to use in the dense layers. Default is 'glorot_uniform'.
    non_linear_proj : str, optional
        Non-linear projection layer to use. Default is None, but you can also use 'glu' for a Gated Linear Unit or
        'selu' for a Scaled Exponential Linear Unit. If non_linear_proj is not None, the output dimension will be twice
        the input dimension.
    contextualized_embeddings : `transformers.TFPreTrainedModel`, optional
        Contextualized embedding model to use, such as BERT or ELMo. If provided, the input tensor will be passed through
        the contextualized embedding model before being processed by the PositionWiseFFN layer.

    Returns
    -------
    output : `tf.Tensor`
        A tensor of shape `(batch size, number of time steps or sequence length in tokens, number of hidden units or
        feature dimension)`.

    Notes
    -----
    The Position-Wise Feed-Forward Layer is a type of feedforward layer consisting of two dense layers that apply to the
    last dimension, which means the same dense layers are used for each position item in the sequence, so called
    position-wise.

    Examples
    --------
    >>> tf.random.set_seed(1)
    >>> ffn = PositionwiseFFN(input_hidden_units=6, output_hidden_units=4, activation='tanh', kernel_initializer='he_uniform', non_linear_proj='glu', contextualized_embeddings=TFBertModel.from_pretrained('bert-base-uncased'))
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
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I.
    (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
    """

    def __init__(
        self,
        input_hidden_units,
        output_hidden_units,
        activation="relu",
        dropout_rate=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer=None,
        non_linear_proj=None,
        contextualized_embeddings=None,
        **kwargs
    ):
        super().__init__(kwargs)
        self.dense1 = tf.keras.layers.Dense(
            input_hidden_units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.non_linear_proj = non_linear_proj
        if self.non_linear_proj == "glu":
            self.glu = tf.keras.layers.Dense(
                output_hidden_units * 2,
                activation="sigmoid",
                kernel_initializer=kernel_initializer,
            )
        elif self.non_linear_proj == "selu":
            print("uner selu sectoin")
            self.selu = tf.keras.layers.Dense(
                output_hidden_units * 2,
                activation="selu",
                kernel_initializer=kernel_initializer,
            )
        else:
            self.dense2 = tf.keras.layers.Dense(
                output_hidden_units,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
        self.contextualized_embeddings = contextualized_embeddings
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x):
        # x.shape: (batch size, number of time steps or sequence length in tokens, number of hidden units or
        # feature dimension)
        if self.contextualized_embeddings is not None:
            bert_output = self.contextualized_embeddings(x)
            x = bert_output[0]
        if self.non_linear_proj == "glu":
            x = self.dense1(x)
            split_units = x.shape[-1] // 2
            x = self.dropout(x)
            return (
                x[:, :, :split_units]
                * tf.keras.activations.sigmoid(self.glu(x[:, :, split_units:]))[
                    :, :, :split_units
                ]
            )
        elif self.non_linear_proj == "selu":
            x = self.dense1(x)
            split_units = x.shape[-1] // 2
            x = self.dropout(x)
            return (
                x[:, :, :split_units]
                * tf.keras.activations.sigmoid(self.selu(x[:, :, split_units:]))[
                    :, :, :split_units
                ]
            )
        else:
            x = self.dropout(x)
            return self.dense2(self.dense1(x))


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == "__main__":
    tf.random.set_seed(1)
    # ffn = PositionWiseFFN(6, 4)
    ffn = PositionwiseFFN(
        input_hidden_units=32,
        output_hidden_units=64,
        activation="tanh",
        kernel_initializer="he_uniform",
        non_linear_proj="glu",
    )
    x = tf.ones((2, 3, 32))
    print(ffn(x))
