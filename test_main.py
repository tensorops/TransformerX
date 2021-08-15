import pytest
from main import MultiHeadAttention
import numpy as np


@pytest.fixture()
def test_transpose_qkv():

    x = np.random.random([100, 10, 5])
    assert MultiHeadAttention.transpose_qkv(x, x)
