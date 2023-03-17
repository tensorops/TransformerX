from typing import Optional, Tuple, Callable

import tensorflow as tf

from transformerx.layers.addnorm import AddNorm
from transformerx.layers.multihead_attention import MultiHeadAttention
from transformerx.layers.positionwise_ffn import PositionWiseFFN


# class TransformerEncoderBlock(tf.keras.layers.Layer):
#     """Transformer encoder block [1]_.
#
#     Include a stack of layers used in the transformer encoder block.
#
#     Parameters
#     ----------
#     d_model : int
#         Dimensions of the queries, keys, and values
#     norm_type :
#         Arbitrary. Shape of the input.
#     ffn_num_hiddens :
#         Number of input hidden units
#     num_heads : int
#         Number of the heads in the multi-head attention
#     dropout_rate :
#         Float between 0 and 1. Fraction of the input units to drop.
#     bias : bool - default = False
#         Indicates the usage of bias in the dense layers (i.e. W_q, W_k, W_v, and W_o)
#     """
#
#     def __init__(
#             self,
#             d_model,
#             ffn_num_hiddens,
#             num_heads,
#             dropout_rate,
#             norm_type='layer',
#             bias=False,
#     ):
#         super().__init__()
#         self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate, bias)
#         self.addnorm1 = AddNorm(norm_type, dropout_rate)
#         self.ffn = PositionWiseFFN(ffn_num_hiddens, d_model)
#         self.addnorm2 = AddNorm(norm_type, dropout_rate)
#
#     def call(self, X, valid_lens, **kwargs):
#         Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
#         return self.addnorm2(Y, self.ffn(Y), **kwargs)

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            d_model: int,  # Dimensionality of the input and output tensors
            ffn_num_hiddens: int,  # Number of hidden units in the feedforward network
            num_heads: int,  # Number of attention heads
            dropout_rate: float,  # Dropout rate for the attention and feedforward networks
            use_norm: bool = True,  # Whether to use normalization (layer or batch)
            residual_connections: Optional[Tuple[bool, bool]] = None,  # Whether to use residual connections
            activation_fn: Optional[Callable] = None,  # Activation function for the feedforward network
            clip_norm: Optional[float] = None,  # Maximum norm for gradient clipping
            kernel_initializer: Optional[Callable] = None,  # Initializer for the kernel weights
            bias_initializer: Optional[Callable] = None,  # Initializer for the bias weights
            mixed_precision: bool = False,  # Whether to use mixed precision training
            learning_rate_schedule: Optional[Callable] = None,  # Learning rate schedule function
            norm_type: str = 'layer',  # Type of normalization (layer or batch)
            bias: bool = False,  # Whether to include bias terms in the attention computation
    ):
        super().__init__()
        assert isinstance(d_model, int) and d_model > 0, 'Invalid d_model: {}'.format(d_model)
        assert isinstance(ffn_num_hiddens, int) and ffn_num_hiddens > 0, 'Invalid ffn_num_hiddens: {}'.format(ffn_num_hiddens)
        assert isinstance(num_heads, int) and num_heads > 0 and d_model//num_heads==0, 'Invalid num_heads: {}'.format(num_heads)
        assert isinstance(dropout_rate, float) and 0.0 <= dropout_rate <= 1.0, 'Invalid dropout rate: {}'.format(dropout_rate)
        assert norm_type in ['layer', 'batch', 'instance'], 'Invalid norm_type: {}'.format(norm_type)
        assert isinstance(bias, bool), 'Invalid bias: {}'.format(bias)
        if residual_connections is not None:
            assert len(residual_connections) == 2, 'residual_connections should be a tuple of two boolean values'
        if activation_fn is not None:
            assert callable(activation_fn), 'activation_fn should be a callable object'
        if clip_norm is not None:
            assert isinstance(clip_norm, float) and clip_norm > 0.0, 'Invalid clip_norm: {}'.format(clip_norm)
        if kernel_initializer is not None:
            assert callable(kernel_initializer), 'kernel_initializer should be a callable object'
        if bias_initializer is not None:
            assert callable(bias_initializer), 'bias_initializer should be a callable object'
        if mixed_precision:
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(policy)
        if learning_rate_schedule is not None:
            assert callable(learning_rate_schedule), 'learning_rate_schedule should be a callable object'

        self.dmodel = d_model
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate, bias)
        self.addnorm1 = AddNorm(norm_type, dropout_rate) if use_norm else None
        self.ffn = PositionWiseFFN(ffn_num_hiddens, d_model, activation_fn, kernel_initializer, bias_initializer)
        self.addnorm2 = AddNorm(norm_type, dropout_rate) if use_norm else None
        self.residual_connections = residual_connections
        self.clip_norm = clip_norm
        self.mixed_precision = mixed_precision
        self.learning_rate_schedule = learning_rate_schedule

    def call(self, X, valid_lens, training=None, **kwargs):
        assert len(X.shape) == 3, 'Input tensor should have rank 3'
        assert X.shape[-1] == self.d_model, 'Last dimension of input tensor should be equal to d_model'
        assert len(valid_lens.shape) == 1, 'valid_lens should be a 1D tensor'
        assert isinstance(valid_lens[0], int), 'Elements of valid_lens should be integers'

        attn_output = self.attention(X, X, X, valid_lens, training=training, **kwargs)
        if self.addnorm1:
            attn_output = self.addnorm1(X, attn_output, training=training, **kwargs)
        ffn_output = self.ffn(attn_output, training=training, **kwargs)
        if self.addnorm2:
            ffn_output = self.addnorm2(attn_output, ffn_output, training=training, **kwargs)
        if self.residual_connections is None:
            output = ffn_output
        else:
            output = ffn_output if self.residual_connections[1] else attn_output
            output = X + output if self.residual_connections[0] else output
        if self.clip_norm is not None:
            output = tf.clip_by_norm(output, self.clip_norm)
        if self.mixed_precision:
            output = tf.keras.mixed_precision.experimental.cast(output, dtype=tf.float32)
        if self.learning_rate_schedule is not None:
            global_step = kwargs.get('global_step', None)
            if global_step is None:
                raise ValueError('global_step must be provided if learning_rate_schedule is not None')
            learning_rate = self.learning_rate_schedule(global_step)
            self.add_metric(learning_rate, name='learning_rate')
        return output

