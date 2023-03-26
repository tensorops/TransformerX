import pytest
import tensorflow as tf
from transformerx.layers import PositionwiseFFN


class TestPositionwiseFFN:
    @pytest.fixture
    def layer(self):
        return PositionwiseFFN(
            input_hidden_units=128,
            output_hidden_units=64,
            activation="relu",
            init="glorot_uniform",
            non_linear_proj="glu",
            contextualized_embeddings=None,
            dropout_rate=0.1,
        )

    def test_layer_output_shape(self, layer):
        input_tensor = tf.random.normal([32, 20, 128])
        output_tensor = layer(input_tensor)
        assert output_tensor.shape == (32, 20, 64)

    def test_layer_output_type(self, layer):
        input_tensor = tf.random.normal([32, 20, 128])
        output_tensor = layer(input_tensor)
        assert isinstance(output_tensor, tf.Tensor)
