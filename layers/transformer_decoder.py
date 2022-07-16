import tensorflow as tf

from layers.positional_encoding import PositionalEncoding
from layers.transformer_decoder_block import TransformerDecoderBlock


class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer decoder that encompasses one or more TransformerDecoderBlock blocks."""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        num_blks,
        dropout,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [
            TransformerDecoderBlock(
                num_hiddens,
                norm_shape,
                ffn_num_hiddens,
                num_heads,
                dropout,
                i,
            )
            for i in range(num_blks)
        ]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def call(self, X, state, **kwargs):
        X = self.pos_encoding(
            self.embedding(X)
            * tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)),
            **kwargs,
        )
        # 2 attention layers in decoder
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
