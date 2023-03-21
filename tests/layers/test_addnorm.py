import numpy as np
import pytest
import tensorflow as tf

from transformerx.layers import AddNorm


class TestAddNorm:
    def test_init(self):
        # Test that the layer initializes correctly
        norm_type = "layer"
        dropout_rate = 0.2
        addnorm = AddNorm(norm_type=norm_type, dropout_rate=dropout_rate)

        assert addnorm.dropout_rate == dropout_rate

        # Test for invalid input for dropout_rate
        norm_type = "batch"
        dropout_rate = 1.2
        with pytest.raises(ValueError):
            addnorm = AddNorm(norm_type=norm_type, dropout_rate=dropout_rate)

        # Test for invalid input type for norm_shape
        norm_type = "instance2"
        dropout_rate = 0.2
        with pytest.raises(TypeError):
            addnorm = AddNorm(norm_type, dropout_rate)

    def test_call(self):
        def test_call():
            norm_shape = [0, 1]
            dropout_rate = 0.2
            addnorm = AddNorm(norm_shape, dropout_rate)

            x = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
            y = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)

            # Test the shape of the output tensor
            assert addnorm(x, y).shape == (5, 2)

            # Test the output tensor values
            expected_output = np.array(
                [
                    [-1.5666986, -1.2185433],
                    [-0.8703881, -0.52223283],
                    [-0.17407762, 0.17407762],
                    [0.52223283, 0.8703881],
                    [1.2185433, 1.5666986],
                ]
            )

            np.testing.assert_almost_equal(addnorm(x, y), expected_output, decimal=4)

    def test_invalid_dropout_rate(self):
        # Test that a ValueError is raised if an invalid dropout rate is provided
        norm_shape = (1, 2)
        dropout_rate = -0.2  # This should be between 0 and 1

        with pytest.raises(ValueError):
            addnorm = AddNorm(norm_shape, dropout_rate)
