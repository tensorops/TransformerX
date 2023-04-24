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

    def test_transformer_encoder_block_with_attention_mask(
        self, transformer_encoder_block
    ):
        input_tensor = tf.random.uniform((32, 10, 512))
        attention_mask = tf.ones((32, 10))
        output_tensor, attn_weights = transformer_encoder_block(
            input_tensor, attention_mask=attention_mask
        )
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_residual_connections(
        self, transformer_encoder_block
    ):
        input_tensor = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.residual_connections = (True, True)
        output_tensor, attn_weights = transformer_encoder_block(input_tensor)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_clip_norm(self, transformer_encoder_block):
        input_tensor = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.clip_norm = 1.0
        output_tensor, attn_weights = transformer_encoder_block(input_tensor)
        assert tf.math.reduce_max(tf.norm(output_tensor, axis=-1)) <= 1.0

    def test_transformer_encoder_block_with_layer_norm(self, transformer_encoder_block):
        input_tensor = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.use_norm = True
        output_tensor, attn_weights = transformer_encoder_block(input_tensor)
        assert output_tensor.shape == (32, 10, 512)
