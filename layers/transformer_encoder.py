import tensorflow as tf

from layers.positional_encoding import PositionalEncoding
from layers.transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder that encompasses one or more TransformerEncoderBlock blocks."""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        num_blks,
        dropout,
        bias=False,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [
            TransformerEncoderBlock(
                num_hiddens,
                norm_shape,
                ffn_num_hiddens,
                num_heads,
                dropout,
                bias,
            )
            for _ in range(num_blks)
        ]

    def call(self, X, valid_lens, **kwargs):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(
            self.embedding(X)
            * tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)),
            **kwargs,
        )
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
