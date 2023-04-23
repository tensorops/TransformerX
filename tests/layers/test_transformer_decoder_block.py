import pytest
import tensorflow as tf
from keras.layers import Layer, Input
from keras.models import Model

from transformerx.layers import TransformerDecoderBlock


# Assuming the necessary imports for the TransformerDecoderBlock and its dependencies are done


def test_transformer_decoder_block():
    batch_size = 2
    seq_length = 10
    queries = tf.random.uniform((batch_size, seq_length, 512))
    keys = tf.random.uniform((batch_size, seq_length, 512))
    values = tf.random.uniform((batch_size, seq_length, 512))
    valid_lens = tf.ones((batch_size, seq_length))

    decoder_block = TransformerDecoderBlock()
    output, attn1_weights, attn2_weights = decoder_block(
        queries, keys, values, valid_lens
    )

    assert output.shape == (batch_size, seq_length, 512)
    assert attn1_weights.shape == (batch_size, 8, seq_length, seq_length)
    assert attn2_weights.shape == (batch_size, 8, seq_length, seq_length)
