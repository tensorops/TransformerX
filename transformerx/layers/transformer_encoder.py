import os
from typing import Optional, Callable, Tuple

import tensorflow as tf

from transformerx.layers.positional_encoding import SinePositionalEncoding
from transformerx.layers.transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder that encompasses one or more TransformerEncoderBlock blocks.

    The TransformerEncoder class provides a Keras layer for the encoder part of the Transformer
    architecture. It consists of an embedding layer, a positional encoding layer, and one or more
    TransformerEncoderBlock blocks. The input sequence is first embedded and then passed through
    the positional encoding layer to obtain the final input representation. The input representation
    is then passed through the TransformerEncoderBlock blocks in a sequential manner to compute
    the final output representation.

    Parameters
    ----------
    vocab_size : int
        The size of the input vocabulary.
    d_model : int
        The d_model of the input and output representations.
    norm_shape : int
        The shape of the normalization layer.
    ffn_num_hiddens : int
        The number of hidden units in the feed-forward network of the TransformerEncoderBlock.
    num_heads : int
        The number of attention heads in the TransformerEncoderBlock.
    n_blocks : int
        The number of TransformerEncoderBlock blocks.
    dropout : float
        The dropout rate.
    use_bias : bool
        Whether to use bias in the linear transformations of the TransformerEncoderBlock.
        Default is False.

    Attributes
    ----------
    embedding : tf.keras.layers.Embedding
        The embedding layer for the input sequence.
    pos_encoding : SinePositionalEncoding
        The positional encoding layer for the input sequence.
    blocks : List[TransformerEncoderBlock]
        The list of TransformerEncoderBlock blocks.
    attention_weights : List[tf.Tensor]
        The list of attention weights tensors computed by each Transformer


    Examples
    --------
    >>> # Create a TransformerEncoder instance with a vocabulary size of 1000, a d_model of 128,
    >>> # a normalization shape of 4, 64 hidden units in the feed-forward network, 8 attention heads,
    >>> # 2 TransformerEncoderBlock blocks, and a dropout rate of 0.1
    >>> transformer_encoder = TransformerEncoder(
    ...     vocab_size=1000,
    ...     d_model=128,
    ...     norm_shape=4,
    ...     ffn_num_hiddens=64,
    ...     num_heads=8,
    ...     n_blocks=2,
    ...     dropout=0.1
    ... )

    >>> # Compute the output representation for a batch of input sequences
    >>> input_sequences = tf.random.uniform((batch_size, seq_length))
    >>> valid_lens = tf.random.uniform((batch_size,))
    >>> output_representation = transformer_encoder(input_sequences, valid_lens)

    >>> # Get the attention weights of the TransformerEncoderBlock blocks
    >>> attention_weights = transformer_encoder.attention_weights
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        n_blocks: int = 6,
        maxlen_position_encoding: int = 10000,
        attention_dropout: float = 0.0,
        norm_type: str = "layer",
        norm_eps: float = 1e-6,
        use_norm: bool = True,
        rescale_embedding: bool = False,
        dropout_rate: float = 0.1,
        attention_mechanism: str = "scaled_dotproduct",
        input_hidden_units_ffn: int = 64,
        residual_connections: Optional[Tuple[bool, bool]] = (True, True),
        activation_fn: Optional[Callable] = tf.nn.relu,
        non_linear_proj=None,
        clip_norm: Optional[float] = 1.0,
        kernel_initializer: Optional[Callable] = tf.keras.initializers.GlorotUniform(),
        bias_initializer: Optional[Callable] = tf.keras.initializers.Zeros(),
        kernel_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = tf.keras.regularizers.l2(0.01),
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        mixed_precision: bool = False,
        learning_rate_schedule: Optional[Callable] = None,
        use_bias: bool = True,
        contextualized_embeddings=None,
        name: str = "TransformerEncoder",
        dtype: Optional[tf.dtypes.DType] = None,
        **kwargs,
    ):
        super(TransformerEncoder, self).__init__(name=name, dtype=dtype, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_blocks = n_blocks
        self.maxlen_position_encoding = maxlen_position_encoding
        self.attention_dropout = attention_dropout
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.use_norm = use_norm
        self.rescale_embedding = rescale_embedding
        self.dropout_rate = dropout_rate
        self.attention_mechanism = attention_mechanism
        self.input_hidden_units_ffn = input_hidden_units_ffn
        self.residual_connections = residual_connections
        self.activation_fn = activation_fn
        self.non_linear_proj = non_linear_proj
        self.clip_norm = clip_norm
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.mixed_precision = mixed_precision
        self.learning_rate_schedule = learning_rate_schedule
        self.use_bias = use_bias
        self.contextualized_embeddings = contextualized_embeddings

        self.pos_encoding = SinePositionalEncoding(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            maximum_position_encoding=self.maxlen_position_encoding,
            **kwargs,
        )
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model)
        self.blocks = [
            TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                norm_type=self.norm_type,
                norm_eps=self.norm_eps,
                use_norm=self.use_norm,
                attention_mechanism=self.attention_mechanism,
                input_hidden_units_ffn=self.input_hidden_units_ffn,
                residual_connections=self.residual_connections,
                activation_fn=self.activation_fn,
                non_linear_proj=self.non_linear_proj,
                clip_norm=self.clip_norm,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                mixed_precision=self.mixed_precision,
                learning_rate_schedule=self.learning_rate_schedule,
                use_bias=self.use_bias,
                contextualized_embeddings=self.contextualized_embeddings,
                dtype=self.dtype,
                **kwargs,
            )
            for _ in range(self.n_blocks)
        ]

    def apply_positional_embedding(self, inputs=None, **kwargs):
        # if tf.shape(inputs)
        embedded_inputs = self.embedding(inputs)
        return self.pos_encoding(
            embedded_inputs
            * tf.math.sqrt(tf.cast(self.d_model, dtype=embedded_inputs.dtype)),
            # **kwargs,
        )

    def call(self, queries, attention_mask=None, **kwargs):
        """Compute the output representation for the input sequence.

        This method computes the output representation for the input sequence by first passing it
        through the embedding layer and the positional encoding layer, and then passing the resulting
        input representation through the TransformerEncoderBlock blocks in a sequential manner.

        Parameters
        ----------
        queries : tf.Tensor
            The input sequence tensor of shape (batch_size, seq_length).
        attention_mask : tf.Tensor
            The tensor of valid sequence lengths of shape (batch_size,) or (batch_size, seq_length).
        **kwargs : dict
            Additional keyword arguments to be passed to the TransformerEncoderBlock blocks.

        Returns
        -------
        output_representation : tf.Tensor
            The output representation tensor of shape (batch_size, seq_length, d_model).

        Examples
        --------
        >>> # Create a TransformerEncoder instance with a vocabulary size of 1000, a d_model of 128,
        >>> # a normalization shape of 4, 64 hidden units in the feed-forward network, 8 attention heads,
        >>> # 2 TransformerEncoderBlock blocks, and a dropout rate of 0.1
        >>> transformer_encoder = TransformerEncoder(
        ...     vocab_size=1000,
        ...     d_model=128,
        ...     norm_shape=4,
        ...     ffn_num_hiddens=64,
        ...     num_heads=8,
        ...     n_blocks=2,
        ...     dropout=0.1
        ... )
        >>>
        >>> # Compute the output representation for a batch of input sequences
        >>> input_sequences = tf.random.uniform((batch_size, seq_length))
        >>> valid_lens = tf.random.uniform((batch_size,))
        >>> output_representation = transformer_encoder(input_sequences, attention_mask)
        >>>
        >>> # Get the attention weights of the TransformerEncoderBlock blocks
        >>> attention_weights = transformer_encoder.attention_weights
        """
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        embeddings = self.apply_positional_embedding(queries, **kwargs)
        self.attention_weights = [None] * len(self.blocks)
        for i, blk in enumerate(self.blocks):
            embeddings, attn_weights = blk(
                embeddings, attention_mask=attention_mask, **kwargs
            )
            self.attention_weights[i] = attn_weights
        return embeddings, self.attention_weights


def main():
    # define the model
    inputs = tf.keras.layers.Input(shape=[10])
    x, _ = TransformerEncoder(vocab_size=64, d_model=512, num_heads=8, n_blocks=6)(
        inputs
    )
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # compile the model
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # train the model
    x_train = tf.random.uniform(shape=(1000, 10), maxval=64, dtype=tf.int32)
    y_train = tf.random.uniform(shape=(1000, 1), maxval=2, dtype=tf.int32)
    model.fit(x_train, y_train, epochs=10, batch_size=32)


if __name__ == "__main__":
    main()
