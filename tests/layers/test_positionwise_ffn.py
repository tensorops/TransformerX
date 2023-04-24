import pytest
import tensorflow as tf
from transformerx.layers import PositionwiseFFN


class TestPositionwiseFFN:
    @pytest.fixture
    def layer(self):
        return PositionwiseFFN(
            input_hidden_units=128,
            activation="relu",
            kernel_initializer="glorot_uniform",
            non_linear_proj="glu",
            contextualized_embeddings=None,
            dropout_rate=0.1,
        )

    @pytest.fixture
    def layer_selu(self):
        return PositionwiseFFN(
            input_hidden_units=128,
            activation="relu",
            kernel_initializer="glorot_uniform",
            non_linear_proj="selu",
            contextualized_embeddings=None,
            dropout_rate=0.1,
        )

    @pytest.fixture
    def layer_no_nonlinear(self):
        return PositionwiseFFN(
            input_hidden_units=128,
            activation="relu",
            kernel_initializer="glorot_uniform",
            non_linear_proj=None,
            contextualized_embeddings=None,
            dropout_rate=0.1,
        )

    def test_layer_output_shape(self, layer):
        input_tensor = tf.random.normal([32, 20, 128])
        output_tensor = layer(input_tensor)
        assert output_tensor.shape == (32, 20, 128)

    def test_layer_output_type(self, layer):
        input_tensor = tf.random.normal([32, 20, 128])
        output_tensor = layer(input_tensor)
        assert isinstance(output_tensor, tf.Tensor)

    def test_layer_glu_non_linear_proj(self, layer):
        input_tensor = tf.random.normal([32, 20, 128])
        output_tensor = layer(input_tensor)
        assert output_tensor.shape == (32, 20, 128)

    def test_layer_selu_non_linear_proj(self, layer_selu):
        input_tensor = tf.random.normal([32, 20, 128])
        output_tensor = layer_selu(input_tensor)
        assert output_tensor.shape == (32, 20, 128)

    def test_layer_no_non_linear_proj(self, layer_no_nonlinear):
        input_tensor = tf.random.normal([32, 20, 128])
        output_tensor = layer_no_nonlinear(input_tensor)
        assert output_tensor.shape == (32, 20, 128)
