from typing import Optional, Tuple, Callable

import tensorflow as tf

from transformerx.layers.addnorm import AddNorm
from transformerx.layers.multihead_attention import MultiHeadAttention
from transformerx.layers.positionwise_ffn import PositionwiseFFN


class TransformerDecoderBlockOld(tf.keras.layers.Layer):
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


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int = 512,  # Dimensionality of the input and output tensors
        num_heads: int = 8,  # Number of attention heads
        dropout_rate: float = 0.1,  # Dropout rate for the attention and feedforward networks
        norm_type: str = "layer",  # Type of normalization (layer or batch) (feedforward networks)
        norm_eps: float = 1e-6,
        attention_mechanism: str = "scaled_dotproduct",
        input_hidden_units_ffn: int = 32,  # Number of input hidden units in the feedforward network
        output_hidden_units_ffn: int = 64,  # Number of output hidden units in the feedforward network
        use_norm: bool = True,  # Whether to use normalization (layer or batch)
        residual_connections: Optional[
            Tuple[bool, bool]
        ] = None,  # Whether to use residual connections
        activation_fn: Optional[
            Tuple[Callable, str]
        ] = "relu",  # Activation function for the feedforward network,
        non_linear_proj=None,  # Non-linear projection for poistionwise feedforward network
        clip_norm: Optional[float] = None,  # Maximum norm for gradient clipping
        kernel_initializer: Optional[
            Callable
        ] = None,  # Initializer for the kernel weights
        bias_initializer: Optional[Callable] = None,  # Initializer for the bias weights
        mixed_precision: bool = False,  # Whether to use mixed precision training
        learning_rate_schedule: Optional[
            Callable
        ] = None,  # Learning rate schedule function
        bias: bool = False,  # Whether to include bias terms in the attention computation
        kernel_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = None,  # kernel regularizer for AddNorm
        bias_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = None,  # bias regularizer for AddNorm
        contextualized_embeddings=None,  # incorporate pre-trained language models such as BERT or GPT-2 into the
        # model (feedforward networks)
        causal_mask: bool = True,  # Whether to use a causal mask
        name=None,  # Name of the layer
        i: int = 0,  # Index of the block
        **kwargs,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.i = i
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.bias = bias
        self.attention_mechanism = attention_mechanism

        # Multi-head attention 1
        self.attention1 = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            bias=self.bias,
            attention=self.attention_mechanism,
            causal_mask=causal_mask,
            **kwargs,
        )
        self.addnorm1 = (
            AddNorm(
                norm_type=norm_type,
                norm_eps=norm_eps,
                dropout_rate=dropout_rate,
                activation=activation_fn,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                **kwargs,
            )
            if use_norm
            else None
        )

        # Multi-head attention 2
        self.attention2 = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            bias=self.bias,
            attention=self.attention_mechanism,
            causal_mask=causal_mask,
            **kwargs,
        )
        self.addnorm2 = (
            AddNorm(
                norm_type=norm_type,
                norm_eps=norm_eps,
                dropout_rate=dropout_rate,
                activation=activation_fn,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                **kwargs,
            )
            if use_norm
            else None
        )

        # Position-wise feedforward network
        self.ffn = PositionwiseFFN(
            input_hidden_units=input_hidden_units_ffn,
            output_hidden_units=output_hidden_units_ffn,
            activation=activation_fn,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            non_linear_proj=non_linear_proj,
            contextualized_embeddings=contextualized_embeddings,
            **kwargs,
        )
        self.addnorm3 = (
            AddNorm(
                norm_type=norm_type,
                norm_eps=norm_eps,
                dropout_rate=dropout_rate,
                activation=activation_fn,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                **kwargs,
            )
            if use_norm
            else None
        )

        self.residual_connections = residual_connections
        self.clip_norm = clip_norm
        self.mixed_precision = mixed_precision
        self.learning_rate_schedule = learning_rate_schedule

    # the call method of the transformer decoder block
    def call(self, queries, keys, values, valid_lens, **kwargs):
        # Multi-head attention 1 (self-attention)
        attn_output1, attn1_weights = self.attention1(
            queries, queries, queries, **kwargs
        )
        if self.addnorm1 is not None:
            attn_output1 = self.addnorm1(queries, attn_output1, **kwargs)
        else:
            attn_output1 = queries + attn_output1

        # Multi-head attention 2 (encoder-decoder --cross-- attention)
        attn2_output, attn2_weights = self.attention2(
            attn_output1, keys, values, **kwargs
        )
        if self.addnorm2 is not None:
            attn2_output = self.addnorm2(attn_output1, attn2_output, **kwargs)
        else:
            attn2_output = attn_output1 + attn2_output

        # Position-wise feedforward network
        ffn_output = self.ffn(attn2_output)
        if self.addnorm3 is not None:
            ffn_output = self.addnorm3(attn2_output, ffn_output, **kwargs)
        else:
            ffn_output = attn2_output + ffn_output
        return ffn_output, attn1_weights, attn2_weights
