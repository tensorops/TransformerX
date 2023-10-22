import pytest
import tensorflow as tf
from transformerx.layers.masks.atomic_sparse_attention import DilatedAttentionMask


class TestDilatedAttentionMask:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dilated_attention_mask = DilatedAttentionMask()

    def test_build_mask(self):
        # Test the build_mask method
        q_len = 5
        k_len = 5
        mask = self.dilated_attention_mask.build_mask(q_len, k_len)

        assert mask.shape == (5, 5), "Mask shape mismatch"

    def test_call_with_3d_input(self):
        # Test the call method with a 3D input tensor
        input_tensor = tf.random.uniform((2, 6, 6))
        mask = self.dilated_attention_mask(input_tensor)

        assert mask.shape == (2, 6, 6), "Mask shape mismatch"

    def test_call_with_2d_input(self):
        # Test the call method with a 2D input tensor
        input_tensor = tf.random.uniform((4, 4))
        mask = self.dilated_attention_mask(input_tensor)

        print(input_tensor)
        print(mask)
        assert mask.shape == (4, 4), "Mask shape mismatch"

    def test_call_with_multihead(self):
        # Test the call method with multihead set to True
        self.dilated_attention_mask.multihead = True
        input_tensor = tf.random.uniform((2, 4, 6))
        mask = self.dilated_attention_mask(input_tensor)

        assert mask.shape == (1, 4, 4), "Multihead mask shape mismatch"
        # Add more assertions here if needed

    def test_call_with_custom_dilation_rate(self):
        # Test the call method with a custom dilation rate
        dilation_rate = 2
        custom_dilated_attention_mask = DilatedAttentionMask(
            dilation_rate=dilation_rate
        )
        input_tensor = tf.random.uniform((2, 4, 6))
        mask = custom_dilated_attention_mask(input_tensor)

        assert mask.shape == (1, 4, 4), "Custom dilation rate mask shape mismatch"
        #
