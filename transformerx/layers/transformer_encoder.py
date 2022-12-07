import tensorflow as tf

from transformerx.layers.positional_encoding import AbsolutePositionalEncoding
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
    depth : int
        The depth of the input and output representations.
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
    bias : bool
        Whether to use bias in the linear transformations of the TransformerEncoderBlock.
        Default is False.

    Attributes
    ----------
    embedding : tf.keras.layers.Embedding
        The embedding layer for the input sequence.
    pos_encoding : AbsolutePositionalEncoding
        The positional encoding layer for the input sequence.
    blocks : List[TransformerEncoderBlock]
        The list of TransformerEncoderBlock blocks.
    attention_weights : List[tf.Tensor]
        The list of attention weights tensors computed by each Transformer


    Examples
    --------
    >>> # Create a TransformerEncoder instance with a vocabulary size of 1000, a depth of 128,
    >>> # a normalization shape of 4, 64 hidden units in the feed-forward network, 8 attention heads,
    >>> # 2 TransformerEncoderBlock blocks, and a dropout rate of 0.1
    >>> transformer_encoder = TransformerEncoder(
    ...     vocab_size=1000,
    ...     depth=128,
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
            vocab_size,
            depth,
            norm_shape,
            ffn_num_hiddens,
            num_heads,
            n_blocks,
            dropout,
            bias=False,
    ):
        super().__init__()
        self.depth = depth
        self.n_blocks = n_blocks
        self.embedding = tf.keras.layers.Embedding(vocab_size, depth)
        self.pos_encoding = AbsolutePositionalEncoding(depth, dropout)
        self.blocks = [
            TransformerEncoderBlock(
                    depth,
                    norm_shape,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    bias,
            )
            for _ in range(self.n_blocks)
        ]

    def call(self, X, valid_lens, **kwargs):
        """Compute the output representation for the input sequence.

        This method computes the output representation for the input sequence by first passing it
        through the embedding layer and the positional encoding layer, and then passing the resulting
        input representation through the TransformerEncoderBlock blocks in a sequential manner.

        Parameters
        ----------
        X : tf.Tensor
            The input sequence tensor of shape (batch_size, seq_length).
        valid_lens : tf.Tensor
            The tensor of valid sequence lengths of shape (batch_size,) or (batch_size, seq_length).
        **kwargs : dict
            Additional keyword arguments to be passed to the TransformerEncoderBlock blocks.

        Returns
        -------
        output_representation : tf.Tensor
            The output representation tensor of shape (batch_size, seq_length, depth).

        Examples
        --------
        >>> # Create a TransformerEncoder instance with a vocabulary size of 1000, a depth of 128,
        >>> # a normalization shape of 4, 64 hidden units in the feed-forward network, 8 attention heads,
        >>> # 2 TransformerEncoderBlock blocks, and a dropout rate of 0.1
        >>> transformer_encoder = TransformerEncoder(
        ...     vocab_size=1000,
        ...     depth=128,
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
        >>> output_representation = transformer_encoder(input_sequences, valid_lens)
        >>>
        >>> # Get the attention weights of the TransformerEncoderBlock blocks
        >>> attention_weights = transformer_encoder.attention_weights
        """
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(
                self.embedding(X) * tf.math.sqrt(tf.cast(self.depth, dtype=tf.float32)),
                **kwargs,
        )
        self.attention_weights = [None] * len(self.blocks)
        for i, blk in enumerate(self.blocks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
