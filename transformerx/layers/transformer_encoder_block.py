from typing import Optional, Tuple, Callable

import tensorflow as tf

from transformerx.layers.addnorm import AddNorm
from transformerx.layers.multihead_attention import MultiHeadAttention
from transformerx.layers.positionwise_ffn import PositionwiseFFN


class TransformerEncoderBlock1(tf.keras.layers.Layer):
    """Transformer encoder block [1]_.

    Include a stack of layers used in the transformer encoder block.

    Parameters
    ----------
    d_model : int
        Dimensions of the queries, keys, and values
    norm_type :
        Arbitrary. Shape of the input.
    ffn_num_hiddens :
        Number of input hidden units
    num_heads : int
        Number of the heads in the multi-head attention
    dropout_rate :
        Float between 0 and 1. Fraction of the input units to drop.
    bias : bool - default = False
        Indicates the usage of bias in the dense layers (i.e. W_q, W_k, W_v, and W_o)
    """

    def __init__(
        self,
        d_model,
        ffn_num_hiddens,
        num_heads,
        dropout_rate,
        norm_type="layer",
        bias=False,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate, bias)
        self.addnorm1 = AddNorm(norm_type, dropout_rate)
        self.ffn = PositionwiseFFN(ffn_num_hiddens, d_model)
        self.addnorm2 = AddNorm(norm_type, dropout_rate)

    def call(self, X, attention_mask, **kwargs):
        Y = self.addnorm1(
            X, self.attention(X, X, X, attention_mask, **kwargs), **kwargs
        )
        return self.addnorm2(Y, self.ffn(Y), **kwargs)


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer Encoder Block

    The TransformerEncoderBlock is a custom layer in TensorFlow Keras that implements a single block of the Transformer
    encoder architecture [1]_, which is a key component of the Transformer model for natural language processing tasks such
    as machine translation, text classification, and language generation. The layer includes self-attention mechanism,
    feedforward networks, and residual connections with optional normalization and other customization options.

    Parameters
    ----------
    d_model: int (default=512)
        Dimensionality of the input and output tensors.
    num_heads: int (default=8)
        Number of attention heads.
    dropout_rate: float (default=0.0)
        Dropout rate for the attention and feedforward networks.
    norm_type: str (default="layer")
        Type of normalization to be applied to the output of the feedforward networks. Can be either "layer", "batch",
        or "instance".
    norm_eps: float (default=1e-6)
        Epsilon value for numerical stability in normalization.
    attention_mechanism: str (default="scaled_dotproduct")
        Type of attention mechanism to be used in the self-attention layer. Currently supports "scaled_dotproduct" and
        other custom attention mechanisms.
    input_hidden_units_ffn: int (default=32)
        Number of hidden units in the input layer of the feedforward networks.
    output_hidden_units_ffn: int (default=64)
        Number of hidden units in the output layer of the feedforward networks.
    use_norm: bool (default=True)
        Whether to apply normalization to the output of the feedforward networks.
    residual_connections: Optional[Tuple[bool, bool]] (default=None)
        Tuple indicating whether to apply residual connections before and after the self-attention and feedforward
        networks. If None, residual connections will be used by default.
    activation_fn: Optional[Callable] (default=None)
        Activation function to be used in the feedforward networks. If None, ReLU activation will be used by default.
    non_linear_proj: Optional (default=None)
        Non-linear projection function to be applied after the self-attention layer. If None, no non-linear projection
        will be applied.
    clip_norm: Optional[float] (default=None)
        Maximum norm for gradient clipping during training.
    kernel_initializer: Optional[Callable] (default=None)
        Initializer for the kernel weights of the self-attention and feedforward networks.
    bias_initializer: Optional[Callable] (default=None)
        Initializer for the bias weights of the self-attention and feedforward networks.
    mixed_precision: bool (default=False)
        Whether to use mixed precision training, which combines float16 and float32 data types for faster training.
    learning_rate_schedule: Optional[Callable] (default=None)
        Learning rate schedule function to be applied during training. If None, no learning rate schedule will be used.
    bias: bool (default=False)
        Whether to include bias terms in the computation of the self-attention weights.
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] (default=None)
        Regularizer for the kernel weights of the AddNorm layer.
    bias_regularizer: Optional[tf.keras.regularizers.Regularizer] (default=None)
        Regularizer for the bias weights of the AddNorm layer.
    contextualized_embeddings: None (default=None)
        Pre-trained language model embeddings to be incorporated into the feedforward networks for contextualization of
        input embeddings.
    **kwargs:
        Additional keyword arguments for the parent class tf.keras.layers.Layer.

    Raises
    ------
    ValueError
        If any of the input arguments are invalid:
        - d_model is not a positive integer.
        - input_hidden_units_ffn is not a positive integer.
        - num_heads is not a positive integer or not divisible by d_model.
        - dropout_rate is not a float between 0 and 1.
        - norm_type is not one of "layer", "batch", or "instance".
        - residual_connections is not None and not a tuple of two boolean values.
        - activation_fn is not callable.
        - clip_norm is not None and not a positive float.
        - kernel_initializer is not callable.
        - bias_initializer is not callable.
        - learning_rate_schedule is not callable.
        - mixed_precision is True but TensorFlow is not configured for mixed precision training.
        - bias is not a boolean.

    Examples
    --------
    >>> # Example 1: Initialize a custom transformer layer with default parameters
    >>> encoder_block = TransformerEncoderBlock()


    >>> # Example 2: Initialize a custom transformer layer with custom parameters
    >>> encoder_block = TransformerEncoderBlock(
    ...     d_model=256,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     norm_type="batch",
    ...     norm_eps=1e-5,
    ...     attention_mechanism="scaled_dotproduct",
    ...     input_hidden_units_ffn=64,
    ...     output_hidden_units_ffn=128,
    ...     use_norm=True,
    ...     residual_connections=(True, True),
    ...     activation_fn=tf.nn.relu,
    ...     non_linear_proj=None,
    ...     clip_norm=1.0,
    ...     kernel_initializer=tf.keras.initializers.GlorotUniform(),
    ...     bias_initializer=tf.keras.initializers.Zeros(),
    ...     mixed_precision=False,
    ...     learning_rate_schedule=None,
    ...     bias=True,
    ...     kernel_regularizer=tf.keras.regularizers.l2(0.01),
    ...     bias_regularizer=None,
    ...     contextualized_embeddings=None
    ... )

    See Also
    --------
    :func:`MultiHeadAttention` : The multi-head self-attention layer used in the TransformerEncoderBlock.
    :func:`PositionwiseFFN` : The position-wise feed-forward network used in the TransformerEncoderBlock.
    :func:`AddNorm` : The residual connection with layer normalization used in the TransformerEncoderBlock.

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L.,
        & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information
        processing systems (pp. 5998-6008).

    Notes
    -----
    - This implementation follows the original Transformer model proposed by Vaswani et al. [1]_.
    - The `attention_mechanism` parameter allows the user to specify the type of attention
      mechanism to use in the multi-head self-attention layer. Possible values are "scaled_dot_product"
      and "masked". If not specified, "scaled_dot_product" is used by default.
    - The `use_norm` parameter controls whether to apply layer normalization after the multi-head
      self-attention and FFN layers. If set to False, no layer normalization is applied.
    - The `contextualized_embeddings` parameter allows the user to specify pre-trained contextualized
      embeddings, such as BERT or ELMo embeddings, to be used in the FFN layer. If not specified,
      standard embeddings are used by default.
    - The `mixed_precision` parameter enables mixed precision training using TensorFlow's
      experimental `mixed_float16` policy.
    - The `learning_rate_schedule` parameter allows the user to specify a custom learning rate
      schedule for the optimizer. The learning rate schedule should be a callable object that takes
      the current training step as input and returns the learning rate for that step.
    """

    def __init__(
        self,
        d_model: int = 512,  # Dimensionality of the input and output tensors
        num_heads: int = 8,  # Number of attention heads
        dropout_rate: float = 0.0,  # Dropout rate for the attention and feedforward networks
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
            Callable
        ] = None,  # Activation function for the feedforward network
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
        contextualized_embeddings=None,
        # incorporate pre-trained language models such as BERT or GPT-2 into the model (feedforward networks)
        **kwargs,
    ):
        super().__init__()
        assert isinstance(d_model, int) and d_model > 0, "Invalid d_model: {}".format(
            d_model
        )
        assert (
            isinstance(input_hidden_units_ffn, int) and input_hidden_units_ffn > 0
        ), "Invalid ffn_num_hiddens: {}".format(input_hidden_units_ffn)
        assert (
            isinstance(num_heads, int) and num_heads > 0 and d_model % num_heads == 0
        ), "Invalid num_heads: {}".format(num_heads)
        assert (
            isinstance(dropout_rate, float) and 0.0 <= dropout_rate <= 1.0
        ), "Invalid dropout rate: {}".format(dropout_rate)
        assert norm_type in [
            "layer",
            "batch",
            "instance",
        ], "Invalid norm_type: {}".format(norm_type)
        assert isinstance(bias, bool), "Invalid bias: {}".format(bias)
        if residual_connections is not None:
            assert (
                len(residual_connections) == 2
            ), "residual_connections should be a tuple of two boolean values"
        if activation_fn is not None:
            assert callable(activation_fn), "activation_fn should be a callable object"
        if clip_norm is not None:
            assert (
                isinstance(clip_norm, float) and clip_norm > 0.0
            ), "Invalid clip_norm: {}".format(clip_norm)
        if kernel_initializer is not None:
            assert callable(
                kernel_initializer
            ), "kernel_initializer should be a callable object"
        if bias_initializer is not None:
            assert callable(
                bias_initializer
            ), "bias_initializer should be a callable object"

        if learning_rate_schedule is not None:
            assert callable(
                learning_rate_schedule
            ), "learning_rate_schedule should be a callable object"

        self.d_model = d_model
        self.attention = MultiHeadAttention(
            d_model, num_heads, dropout_rate, bias, attention_mechanism, **kwargs
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
        self.ffn = PositionwiseFFN(
            input_hidden_units=input_hidden_units_ffn,
            # output_hidden_units=output_hidden_units_ffn,
            activation=activation_fn,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            non_linear_proj=non_linear_proj,
            contextualized_embeddings=contextualized_embeddings,
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
        self.residual_connections = residual_connections
        self.clip_norm = clip_norm
        self.mixed_precision = mixed_precision
        self.learning_rate_schedule = learning_rate_schedule

        if self.mixed_precision:
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)

    def call(self, X, attention_mask=None, training=None, **kwargs):
        assert len(X.shape) == 3, "Input tensor should have rank 3"
        assert (
            X.shape[-1] == self.d_model
        ), "Last dimension of input tensor should be equal to d_model"
        # if attention_mask is not None:
        # attention_mask = tf.cast(attention_mask, tf.int32)
        # assert (
        #     len(attention_mask.shape) == 1
        # ), "attention_mask should be a 1D tensor"
        # assert isinstance(attention_mask[0].numpy(), int), 'Elements of attention_mask should be integers'

        attn_output, attn_weights = self.attention(
            X, X, X, attention_mask, training=training, **kwargs
        )
        if self.addnorm1:
            attn_output = self.addnorm1(X, attn_output, training=training, **kwargs)
        ffn_output = self.ffn(attn_output, training=training, **kwargs)
        if self.addnorm2:
            ffn_output = self.addnorm2(
                attn_output, ffn_output, training=training, **kwargs
            )
        if self.residual_connections is None:
            output = ffn_output
        else:
            output = ffn_output if self.residual_connections[1] else attn_output
            output = X + output if self.residual_connections[0] else output
        if self.clip_norm is not None:
            output = tf.clip_by_norm(output, self.clip_norm)
        if self.learning_rate_schedule is not None:
            global_step = kwargs.get("global_step", None)
            if global_step is None:
                raise ValueError(
                    "global_step must be provided if learning_rate_schedule is not None"
                )
            learning_rate = self.learning_rate_schedule(global_step)
            self.add_metric(learning_rate, name="learning_rate")
        return output, attn_weights


def main():
    # Set up a dummy batch of inputs and attention_mask
    batch_size = 4
    seq_length = 32
    d_model = 512
    inputs = tf.random.normal((batch_size, seq_length, d_model))
    # attention_mask = tf.constant([10, 8, 6, 10], dtype=tf.int32)
    # attention_mask = tf.cast(attention_mask, tf.int32)
    # attention_mask = tf.random.uniform((32, 1, 1, 64), maxval=2, dtype=tf.float32)
    # attention_mask = tf.zeros((seq_length), dtype=tf.int32)
    attention_mask = tf.ones((batch_size, seq_length, seq_length), dtype=tf.bool)
    p = 0.5
    attention_mask = tf.random.uniform((batch_size, seq_length, seq_length)) < p
    attention_mask = tf.cast(attention_mask, dtype=tf.bool)
    # Initialize a TransformerEncoderBlock object
    encoder_block = TransformerEncoderBlock(
        d_model=512,
        num_heads=4,
        dropout_rate=0.1,
        norm_type="batch",
        norm_eps=1e-5,
        attention_mechanism="scaled_dotproduct",
        input_hidden_units_ffn=64,
        output_hidden_units_ffn=128,
        use_norm=True,
        residual_connections=(True, True),
        activation_fn=tf.nn.relu,
        non_linear_proj=None,
        clip_norm=1.0,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
        bias_initializer=tf.keras.initializers.Zeros(),
        mixed_precision=False,
        learning_rate_schedule=None,
        bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        bias_regularizer=None,
        contextualized_embeddings=None,
    )
    encoder_block2 = TransformerEncoderBlock(
        d_model=d_model, num_heads=d_model // seq_length, dropout_rate=0.1
    )

    # Pass the inputs through the encoder block
    outputs = encoder_block(inputs, attention_mask=attention_mask)

    # Check that the output tensor has the correct shape
    assert outputs.shape == (
        batch_size,
        seq_length,
        d_model,
    ), f"{outputs.shape}, ({batch_size}, {seq_length})"

    print("Test passed.")


if __name__ == "__main__":
    main()
