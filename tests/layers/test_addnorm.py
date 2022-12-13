import numpy as np
import pytest
import tensorflow as tf

from transformerx.layers import AddNorm


class TestAddNorm:
    def test_init(self):
        # Test that the layer initializes correctly
        norm_shape = (1, 2)
        dropout_rate = 0.2
        addnorm = AddNorm(norm_shape, dropout_rate)

        assert addnorm.norm_shape == norm_shape
        assert addnorm.dropout_rate == dropout_rate

    def test_invalid_dropout_rate(self):
        # Test that a ValueError is raised if an invalid dropout rate is provided
        norm_shape = (1, 2)
        dropout_rate = -0.2  # This should be between 0 and 1

        with pytest.raises(ValueError):
            addnorm = AddNorm(norm_shape, dropout_rate)
