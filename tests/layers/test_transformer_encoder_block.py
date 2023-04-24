import tensorflow as tf
import pytest
from transformerx.layers import TransformerEncoderBlock


class TestTransformerEncoderBlock:
    @pytest.fixture
    def transformer_encoder_block(self):
        return TransformerEncoderBlock()
