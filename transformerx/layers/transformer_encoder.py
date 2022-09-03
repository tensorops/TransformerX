import tensorflow as tf

from transformerx.layers.positional_encoding import PositionalEncoding
from transformerx.layers.transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder that encompasses one or more TransformerEncoderBlock blocks."""

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
        self.pos_encoding = PositionalEncoding(depth, dropout)
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
