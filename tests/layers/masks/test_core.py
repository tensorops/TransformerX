import tensorflow as tf
from transformerx.layers import DotProductAttention, MultiHeadAttention
from transformerx.layers.masks import PaddingMask, LookAheadMask


def test_lookahead_mask():
    q_input_tensor = tf.random.uniform((2, 4, 6))
    attn_o, attn_w = DotProductAttention()(
        q_input_tensor, q_input_tensor, q_input_tensor
    )

    la_mask = LookAheadMask()
    output_tensor = la_mask(attn_w)

    assert output_tensor.shape == attn_w.shape
    assert tf.reduce_all(
        output_tensor == tf.linalg.band_part(tf.ones_like(attn_w), -1, 0)
    )


def test_multihead_attention():
    q_input_tensor = tf.random.uniform((2, 4, 6))
    input_tensor = tf.random.uniform((2, 4, 6))
    multihead_attn = MultiHeadAttention(d_model=32, num_heads=4, dropout_rate=0.1)
    output, attn_w = multihead_attn(q_input_tensor, input_tensor, input_tensor)

    assert output.shape == q_input_tensor.shape
    assert attn_w.shape == (2, 4, 4)


def test_padding_mask():
    inputs = tf.random.normal(
        (2, 3, 4)
    )  # 3D input tensor (batch_size=2, query_len=3, key_dim=4)
    valid_lens = tf.constant(
        [2, 3], dtype=tf.int32
    )  # Valid lengths for each sequence in the batch

    padding_mask_layer = PaddingMask(multihead=False)

    masked_inputs = padding_mask_layer(inputs, query_len=3, valid_lens=valid_lens)

    assert masked_inputs.shape == inputs.shape


def test_padding_mask_padding_mask_option():
    padding_mask_layer = PaddingMask(multihead=False)

    padding_mask = tf.constant(
        [[False, False, True, True], [False, False, False, True]]
    )  # Example padding mask
    inputs = tf.random.normal((2, 3, 4))
    masked_inputs = padding_mask_layer(
        inputs, padding_mask=padding_mask, query_len=3, key_len=4
    )

    assert masked_inputs.shape == inputs.shape


def test_padding_mask_inputs_option():
    padding_mask_layer = PaddingMask(multihead=False)

    # Test with 3D tensor
    inputs_3d = tf.random.normal((2, 3, 4))
    masked_inputs_3d = padding_mask_layer(inputs=inputs_3d)

    assert masked_inputs_3d.shape == inputs_3d.shape

    # Test with 4D tensor
    inputs_4d = tf.random.normal((2, 3, 4, 5))
    masked_inputs_4d = padding_mask_layer(inputs=inputs_4d)

    assert masked_inputs_4d.shape == inputs_4d.shape


def test_padding_mask_valid_lens_option():
    padding_mask_layer = PaddingMask(multihead=False)

    inputs = tf.random.normal(
        (2, 3, 4)
    )  # 3D input tensor (batch_size=2, query_len=3, key_dim=4)
    valid_lens = tf.constant(
        [2, 3], dtype=tf.int32
    )  # Valid lengths for each sequence in the batch

    masked_inputs = padding_mask_layer(inputs, query_len=3, valid_lens=valid_lens)

    assert masked_inputs.shape == inputs.shape


def test_padding_mask_edge_cases():
    padding_mask_layer = PaddingMask(multihead=False)

    # Test with empty inputs
    inputs = tf.constant([], dtype=tf.float32)
    masked_inputs = padding_mask_layer(
        inputs, query_len=0, valid_lens=tf.constant([], dtype=tf.int32)
    )
    assert tf.equal(masked_inputs, inputs)

    # Test with empty valid_lens
    inputs = tf.random.normal((2, 3, 4))
    valid_lens = tf.constant([], dtype=tf.int32)
    masked_inputs = padding_mask_layer(inputs, query_len=3, valid_lens=valid_lens)
    assert tf.equal(masked_inputs, inputs)


def test_padding_mask_advanced_cases():
    padding_mask_layer = PaddingMask(multihead=False)

    # Test with different padding values
    inputs = tf.constant(
        [[[0, 0, 0], [0, 1, 2]], [[0, 0, 0], [3, 4, 5]]],
        dtype=tf.float32,
    )
    padding_mask = tf.constant([[True, False], [False, True]])
    masked_inputs = padding_mask_layer(
        inputs, padding_mask=padding_mask, query_len=2, key_len=3
    )
    assert tf.reduce_all(
        tf.math.equal(
            masked_inputs,
            tf.constant(
                [[[0, 1, 2], [0, 1, 2]], [[3, 4, 5], [3, 4, 5]]], dtype=inputs.dtype
            ),
        )
    )

    # Test with uneven valid_lens
    inputs = tf.random.normal((3, 4, 5))
    valid_lens = tf.constant([2, 3, 4], dtype=inputs.dtype)
    masked_inputs = padding_mask_layer(inputs, query_len=4, valid_lens=valid_lens)
    assert masked_inputs.shape == inputs.shape

    # Test with large tensor shapes
    inputs = tf.random.normal((10, 20, 30))
    valid_lens = tf.constant(
        [15, 18, 20, 17, 16, 19, 20, 14, 16, 13], dtype=inputs.dtype
    )
    masked_inputs = padding_mask_layer(inputs, query_len=4, valid_lens=valid_lens)
    assert masked_inputs.shape == inputs.shape
