import pytest
import tensorflow as tf
from transformerx.layers import PositionwiseFFN


@pytest.fixture
def layer():
    return PositionwiseFFN(
        input_hidden_units=128,
        output_hidden_units=64,
        activation="relu",
        init="glorot_uniform",
        non_linear_proj="glu",
        contextualized_embeddings=None,
        dropout_rate=0.1,
    )


def test_layer_output_shape(layer):
    input_tensor = tf.random.normal([32, 20, 128])
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (32, 20, 64)


def test_layer_output_type(layer):
    input_tensor = tf.random.normal([32, 20, 128])
    output_tensor = layer(input_tensor)
    assert isinstance(output_tensor, tf.Tensor)


def test_layer_glu_non_linear_proj(layer):
    input_tensor = tf.random.normal([32, 20, 128])
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (32, 20, 64)


def test_layer_selu_non_linear_proj(layer):
    layer.non_linear_proj = "selu"
    input_tensor = tf.random.normal([32, 20, 128])
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (32, 20, 64)


def test_layer_no_non_linear_proj(layer):
    layer.non_linear_proj = None
    input_tensor = tf.random.normal([32, 20, 128])
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (32, 20, 64)
