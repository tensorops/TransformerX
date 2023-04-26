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
    """Transformer decoder block [1]_.

    The TransformerDecoderBlock is a custom layer in TensorFlow Keras that implements a single block of the Transformer
    decoder architecture [1]_, which is a key component of the Transformer model for natural language processing tasks
    such as machine translation, text summarization, and language generation. The layer includes multi-head attention
    mechanism, feedforward networks, and residual connections with optional normalization and other customization options.

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
    casual_mask: bool (default=True)
        Weather to use a casual mask. Allows information from the past to be used when computing the attention weights.
    name: str (default=None)
        The name of the layer.
    i: int (default=0)
        Index of the block.
    **kwargs:
        Additional keyword arguments for the parent class tf.keras.layers.Layer.

    Methods
    -------
    call(X, state, **kwargs) :
        Performs the forward propagation of the decoder block

    Examples
    --------
    >>> # Example 1: Initialize a custom transformer layer with default parameters
    >>> decoder_block = TransformerDecoderBlock()

    >>> # Example 2: Initialize a custom transformer layer with custom parameters
    >>> decoder_block = TransformerDecoderBlock(
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
    ...     contextualized_embeddings=None,
    ...     casual_mask=True,
    ...     name=None,
    ...     i=0
    ... )

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
    - The `casual_mask` parameter is useful for tasks where the output at time step t should only depend
      on the inputs up to time step t-1, such as language modeling or sequence prediction.
    """

    def __init__(
        self,
        d_model: int = 512,  # Dimensionality of the input and output tensors
        num_heads: int = 8,  # Number of attention heads
        dropout_rate: float = 0.1,  # Dropout rate for the attention and feedforward networks
        norm_type: str = "layer",  # Type of normalization (layer or batch) (feedforward networks)
        norm_eps: float = 1e-6,
        attention_mechanism: str = "scaled_dotproduct",
        input_hidden_units_ffn: int = 2048,  # Number of input hidden units in the feedforward network
        # output_hidden_units_ffn: int = 512,  # Number of output hidden units in the feedforward network
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
        kernel_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = None,  # kernel regularizer for AddNorm
        bias_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = None,  # bias regularizer for AddNorm
        mixed_precision: bool = False,  # Whether to use mixed precision training
        learning_rate_schedule: Optional[
            Callable
        ] = None,  # Learning rate schedule function
        bias: bool = False,  # Whether to include bias terms in the attention computation
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
            # output_hidden_units=output_hidden_units_ffn,
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

        if self.mixed_precision:
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)

    # the call method of the transformer decoder block
    def call(self, queries, keys, values, attention_mask=None, **kwargs):
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

        if self.clip_norm is not None:
            ffn_output = tf.clip_by_norm(ffn_output, self.clip_norm)

        if self.learning_rate_schedule is not None:
            global_step = kwargs.get("global_step", None)
            if global_step is None:
                raise ValueError(
                    "global_step must be provided if learning_rate_schedule is not None"
                )
            learning_rate = self.learning_rate_schedule(global_step)
            self.add_metric(learning_rate, name="learning_rate")

        return ffn_output, attn1_weights, attn2_weights
