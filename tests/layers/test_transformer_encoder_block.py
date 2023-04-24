import tensorflow as tf
import pytest
from transformerx.layers import TransformerEncoderBlock


class TestTransformerEncoderBlock:
    @pytest.fixture
    def transformer_encoder_block(self):
        return TransformerEncoderBlock()

    def test_transformer_encoder_block_output_shape(self, transformer_encoder_block):
        input_tensor = tf.random.uniform((32, 10, 512))
        output_tensor, attn_weights = transformer_encoder_block(input_tensor)
        assert output_tensor.shape == (32, 10, 512)
