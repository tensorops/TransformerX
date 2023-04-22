import tensorflow as tf

from transformerx.layers.addnorm import AddNorm
from transformerx.layers.multihead_attention import MultiHeadAttention
from transformerx.layers.positionwise_ffn import PositionwiseFFN


class TransformerDecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block [1]_.

    Include a stack of layers used in the transformer decoder block.

    Parameters
    ----------
    num_hiddens :
        Dimensions of the queries, keys, and values
    norm_shape :
        Arbitrary. Shape of the input.
    ffn_num_hiddens :
        Number of input hidden units
    num_heads :
        Number of the heads in the multi-head attention
    dropout_rate :
        Float between 0 and 1. Fraction of the input units to drop.
    i :
        Index of the block
    """

    def __init__(
        self,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        dropout_rate,
        i,
    ):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout_rate)
        self.addnorm1 = AddNorm(norm_shape, dropout_rate)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout_rate)
        self.addnorm2 = AddNorm(norm_shape, dropout_rate)
        self.ffn = PositionwiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout_rate)

    def call(self, X, state, **kwargs):
        """Forward propagation of the decoder block.

        During training, all the tokens of any output sequence are processed at the same time, so state[2][self.i]
        is None as initialized. When decoding any output sequence token by token during prediction, state[2][self.i]
        contains representations of the decoded output at the i-th block up to the current time step
        """
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1), shape=(-1, num_steps)),
                repeats=batch_size,
                axis=0,
            )
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, depth)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state


class TransformerDecoderBlock1(tf.keras.layers.Layer):
    def __init__(
        self,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        dropout_rate,
        i,
        use_masking=True,
        layer_norm_eps=1e-6,
        residual_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        ffn_activation="relu",
        name=None,
    ):
        super().__init__(name=name)
        self.i = i

        # Multi-head attention 1
        self.attention1 = MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout_rate=attention_dropout_rate,
            causal_mask=use_masking,
            name=f"multi_head_attention_1_{i}",
        )
        self.addnorm1 = AddNorm(
            norm_shape=norm_shape,
            dropout_rate=residual_dropout_rate,
            eps=layer_norm_eps,
            name=f"add_norm_1_{i}",
        )

        # Multi-head attention 2
        self.attention2 = MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout_rate=attention_dropout_rate,
            use_masking=use_masking,
            name=f"multi_head_attention_2_{i}",
        )
        self.addnorm2 = AddNorm(
            norm_shape=norm_shape,
            dropout_rate=residual_dropout_rate,
            eps=layer_norm_eps,
            name=f"add_norm_2_{i}",
        )

        # Position-wise feedforward network
        self.ffn = PositionwiseFFN(
            hidden_size=ffn_num_hiddens,
            output_size=num_hiddens,
            dropout_rate=residual_dropout_rate,
            activation=ffn_activation,
            name=f"positionwise_ffn_{i}",
        )
        self.addnorm3 = AddNorm(
            norm_shape=norm_shape,
            dropout_rate=residual_dropout_rate,
            eps=layer_norm_eps,
            name=f"add_norm_3_{i}",
        )
