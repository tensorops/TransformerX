import pytest
import tensorflow as tf
import numpy as np

from transformerx.layers import TransformerEncoder


class TestTransformerEncoder:
    @pytest.fixture(scope="class")
    def encoder(self):
        return TransformerEncoder(
            vocab_size=1000, max_len=50, d_model=128, num_heads=4, n_blocks=2
        )

    def test_embedding_output_shape(self, encoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        embedded_data = encoder.embedding(input_data)
        assert embedded_data.shape == (2, 3, 128)
