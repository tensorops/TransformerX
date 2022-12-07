import tensorflow as tf

from transformerx.layers.positional_encoding import AbsolutePositionalEncoding
from transformerx.layers.transformer_decoder_block import TransformerDecoderBlock


class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer decoder that encompasses one or more TransformerDecoderBlock blocks.

    Transformer decoder that encompasses one or more TransformerDecoderBlock blocks.

    The TransformerDecoder processes the input sequences and produces the output sequences.
    It also generates attention weights for each of the two attention layers in each of the
    TransformerDecoderBlock blocks.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size. The size of the vocabulary used by the Transformer decoder. This is used to determine the
        size of the input tensor and the output tensor of the call method.
    depth : int
        Dimension of each input sequence. The depth of the input tensor and the output tensor of the call method.
        This is also the size of the hidden states of the Transformer decoder blocks.
    norm_shape : str
        Shape of the normalization layers in the Transformer decoder blocks. This is a tuple of three integers,
        specifying the number of dimensions to normalize over for the first, second, and third normalization layers,
        respectively.
    ffn_num_hiddens : int
        Number of hidden units in the feed-forward neural network layers of the Transformer decoder blocks.
    num_heads : int
        Number of attention heads of the Transformer decoder blocks.
    n_blocks : int
        Number of encoder blocks in the Transformer decoder.
    dropout : float
        The dropout rate to use for the dropout layers in the Transformer decoder blocks. This value is used to
        determine the strength of regularization. A higher dropout rate means that more nodes are dropped out during
        training, which can help prevent overfitting.

    Attributes
    ----------
    depth : int
        The depth (i.e., number of channels) of the input and output representations.
    n_blocks : int
        The number of TransformerDecoderBlock blocks in the decoder.
    embedding : tf.keras.layers.Embedding
        The embedding layer that maps the input sequences to their input representations.
    pos_encoding : AbsolutePositionalEncoding
        The absolute positional encoding layer that adds position information to the input
        representations.
    blocks : List[TransformerDecoderBlock]
        The list of TransformerDecoderBlock blocks that process the input sequences.
    dense : tf.keras.layers.Dense
        The dense layer that maps the output representations to the output sequences.
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
    ):
        super().__init__()
        self.depth = depth
        self.n_blocks = n_blocks
        self.embedding = tf.keras.layers.Embedding(vocab_size, depth)
        self.pos_encoding = AbsolutePositionalEncoding(depth, dropout)
        self.blocks = [
            TransformerDecoderBlock(
                    depth,
                    norm_shape,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    i,
            )
            for i in range(n_blocks)
        ]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.n_blocks]

    def call(self, X, state, **kwargs):
        """Forward call of the Transformer decoder.

        Parameters
        ----------
        X : tf.Tensor
            Input tensor with shape (batch_size, no. of queries or key-value pairs, depth). This tensor is used to
            compute the output of the Transformer decoder.
        state : List
            List of decoder state tensors. This is a list of three elements:
                - enc_outputs: The outputs of the Transformer encoder, of shape (batch_size, max_seq_len, depth).
                This tensor is used to compute the attention weights for the encoder-decoder attention layer in the
                Transformer decoder blocks.
                - enc_valid_lens: The valid lengths of the input sequence to the Transformer encoder, of shape
                (batch_size,) or (batch_size, max_seq_len). This tensor is used to mask the outputs of the Transformer
                encoder so that only the valid parts of the sequence are used in the attention computations.
                - state: A list of length n_blocks, where each element is the state of the corresponding Transformer
                decoder block. This state is used to carry over information from previous time steps when computing the
                output of the Transformer decoder blocks.
        **kwargs
            Additional keyword arguments that can be passed to the Transformer decoder blocks. This can include
            arguments such as the attention mask and the mask for the masked language model.

        Returns
        -------
        output : tf.Tensor
            Output tensor with shape (batch_size, no. of queries or key-value pairs, depth).
        state : List
            Updated list of decoder state tensors.

        Examples
        --------
        >>> # Initialize a TransformerDecoder with a vocabulary size of 1000 and a depth of 512
        >>> decoder = TransformerDecoder(
        ...    vocab_size=1000,
        ...    depth=512,
        ...    norm_shape=(512,),
        ...    n_blocks=6,
        ...    dropout=0.2
        >>> )

        >>> # Define the input sequence, with batch size of 2 and sequence length of 10
        >>> inputs = tf.random.uniform((2, 10), minval=0, maxval=1000, dtype=tf.int32)

        >>> # Define the initial state of the decoder, which should be the output from the encoder
        >>> # and the valid sequence lengths from the encoder
        >>> enc_outputs = tf.random.normal((2, 10, 512))
        >>> enc_valid_lens = tf.constant([10, 5], dtype=tf.int32)
        >>> initial_state = decoder.init_state(enc_outputs, enc_valid_lens)

        >>> # Call the decoder to get the output and the updated state
        >>> output, state = decoder(inputs, initial_state)

        >>> # The output will be a tensor of shape (2, 10, 1000) containing the predicted
        >>> # probabilities for each of the input tokens at each timestep
        >>> # The state will be a tuple containing the encoder outputs, the encoder valid lengths,
        >>> # and a list of attention weights for each block in the decoder
        """

        X = self.pos_encoding(
                self.embedding(X) * tf.math.sqrt(tf.cast(self.depth, dtype=tf.float32)),
                **kwargs,
        )
        # 2 attention layers in decoder
        self._attention_weights = [[None] * len(self.blocks) for _ in range(2)]
        for i, blk in enumerate(self.blocks):
            X, state = blk(X, state, **kwargs)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
