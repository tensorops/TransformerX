import tensorflow as tf
from transformerx.layers import DotProductAttention, MultiHeadAttention
from transformerx.layers.masks import PaddingMask, LookAheadMask


def test_lookahead_mask():
    q_input_tensor = tf.random.uniform((2, 4, 6))
    attn_o, attn_w = DotProductAttention()(
        q_input_tensor, q_input_tensor, q_input_tensor
    )
    mask_value = -1e9
    la_mask = LookAheadMask()
    output_tensor = la_mask(attn_w)
    print(output_tensor)

    # Create a test mask manually using tf.linalg.band_part: this creates upper triangular boolean matrix which then
    # will be multiplied by the mask_value which is 10-9, and then we add this new mask matrix which its upper triangle
    # values are 10-9 with the scores (attention) matrix. Later when we pass this matrix to the softmax, they will
    # become 0 since they are -inf numbers.
    test_mask = 1 - tf.linalg.band_part(tf.ones_like(attn_w), -1, 0)
    test_mask = mask_value * tf.cast(test_mask, dtype=attn_w.dtype)
    test_masked_attn = tf.add(attn_w, test_mask)

    assert output_tensor.shape == attn_w.shape
    tf.debugging.assert_equal(output_tensor, test_masked_attn,
                              message="Mismatch in tensors")
    assert tf.reduce_all(
        output_tensor == test_masked_attn
    )


def test_multihead_attention():
    d_model = 256
    q_input_tensor = tf.random.uniform((2, 4, d_model))
    input_tensor = tf.random.uniform((2, 4, d_model))
    multihead_attn = MultiHeadAttention(d_model=d_model, num_heads=4, dropout_rate=0.1, causal_mask=True)
    output, attn_w = multihead_attn(q_input_tensor, input_tensor, input_tensor)

    assert output.shape == q_input_tensor.shape  # (batch_size, sequence_length, d_model)
    assert attn_w.shape == (2, 4, 4, 4)  # (batch_size, num_heads, sequence_length, sequence_length)


def test_padding_mask():
    # tf.random.uniform((2, 4, 6))
    # # Test with valid lengths
    # scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)
    # valid_lens = tf.constant([3, 2])
    #
    # padding_mask = PaddingMask()
    # masked = padding_mask(scores, valid_lens=valid_lens, input_shape=scores.shape)
    #
    # expected = tf.constant([[1, 2, 3, -1e9], [4, 5, -1e9, -1e9]])
    #
    # assert tf.reduce_all(tf.equal(masked, expected))
    #
    # # Test with padding values
    # scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)
    #
    # padding_mask2 = PaddingMask(padding_value=0)
    # masked = padding_mask2(scores)
    #
    # expected = tf.constant([[1, 2, 3, -1e9], [4, 5, -1e9, -1e9]])
    #
    # assert tf.reduce_all(tf.equal(masked, expected))
    #
    # # Test with padding values
    # scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)
    #
    # padding_mask3 = PaddingMask(padding_value=1)
    # masked = padding_mask3(scores)
    #
    # expected = tf.constant([[-1e9, 2, 3, 0], [4, 5, 0, 0]])
    # print("here is the padding 1: ", masked)
    #
    # assert tf.reduce_all(tf.equal(masked, expected))
    #
    # # Test with multi-head
    # scores = tf.constant(
    #     [[[1, 2, 3, 0], [4, 5, 0, 0]], [[1, 2, 0, 0], [4, 5, 6, 0]]], dtype=tf.float32
    # )
    #
    # print("All tests passed!")

    # Test with padding mask
    padding_mask_values = tf.constant([[0, 0, 1, 1], [0, 1, 1, 1]])
    scores = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.float32)

    padding_mask = PaddingMask()
    masked = padding_mask(scores, padding_mask=padding_mask_values)

    expected = tf.constant([[1, 2, -1e9, -1e9], [5, -1e9, -1e9, -1e9]])

    print(masked)
    assert tf.reduce_all(tf.equal(masked, expected))

    # Test with scores
    scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)

    padding_mask = PaddingMask(padding_value=0)
    masked = padding_mask(scores=scores)

    expected = tf.constant([[1, 2, 3, -1e9], [4, 5, -1e9, -1e9]])

    assert tf.reduce_all(tf.equal(masked, expected))



# def test_padding_mask_padding_mask_option():
#     padding_mask_layer = PaddingMask(multihead=False)
#
#     padding_mask = tf.constant(
#         [[False, False, True, True], [False, False, False, True]]
#     )  # Example padding mask
#     inputs = tf.random.normal((2, 3, 4))
#     masked_inputs = padding_mask_layer(
#         inputs, padding_mask=padding_mask, query_len=3, key_len=4
#     )
#
#     assert masked_inputs.shape == inputs.shape


def test_padding_mask_inputs_option():
    padding_mask_layer = PaddingMask(multihead=False)

    # Test with 3D tensor
    inputs_3d = tf.random.normal((2, 3, 4))
    masked_inputs_3d = padding_mask_layer(scores=inputs_3d)

    assert masked_inputs_3d.shape == inputs_3d.shape

    # Test with 4D tensor
    inputs_4d = tf.random.normal((2, 3, 4, 5))
    masked_inputs_4d = padding_mask_layer(scores=inputs_4d)

    assert masked_inputs_4d.shape == inputs_4d.shape


# def test_padding_mask_valid_lens_option():
#     padding_mask_layer = PaddingMask(multihead=False)
#
#     inputs = tf.random.normal(
#         (2, 3, 4)
#     )  # 3D input tensor (batch_size=2, query_len=3, key_dim=4)
#     valid_lens = tf.constant(
#         [2, 3], dtype=tf.int32
#     )  # Valid lengths for each sequence in the batch
#
#     masked_inputs = padding_mask_layer(inputs, query_len=3, valid_lens=valid_lens)
#
#     assert masked_inputs.shape == inputs.shape


# def test_padding_mask_edge_cases():
#     padding_mask_layer = PaddingMask(multihead=False)
#
#     # Test with empty inputs
#     inputs = tf.constant([], dtype=tf.float32)
#     masked_inputs = padding_mask_layer(
#         inputs, query_len=0, valid_lens=tf.constant([], dtype=tf.int32)
#     )
#     assert tf.equal(masked_inputs, inputs)
#
#     # Test with empty valid_lens
#     inputs = tf.random.normal((2, 3, 4))
#     valid_lens = tf.constant([], dtype=tf.int32)
#     masked_inputs = padding_mask_layer(inputs, query_len=3, valid_lens=valid_lens)
#     assert tf.equal(masked_inputs, inputs)


# def test_padding_mask_advanced_cases():
#     padding_mask_layer = PaddingMask(multihead=False)
#
#     # Test with different padding values
#     inputs = tf.constant(
#         [[[0, 0, 0], [0, 1, 2]], [[0, 0, 0], [3, 4, 5]]],
#         dtype=tf.float32,
#     )
#     padding_mask = tf.constant([[True, False], [False, True]])
#     masked_inputs = padding_mask_layer(
#         inputs, padding_mask=padding_mask, query_len=2, key_len=3
#     )
#     assert tf.reduce_all(
#         tf.math.equal(
#             masked_inputs,
#             tf.constant(
#                 [[[0, 1, 2], [0, 1, 2]], [[3, 4, 5], [3, 4, 5]]], dtype=inputs.dtype
#             ),
#         )
#     )
#
#     # Test with uneven valid_lens
#     inputs = tf.random.normal((3, 4, 5))
#     valid_lens = tf.constant([2, 3, 4], dtype=inputs.dtype)
#     masked_inputs = padding_mask_layer(inputs, query_len=4, valid_lens=valid_lens)
#     assert masked_inputs.shape == inputs.shape
#
#     # Test with large tensor shapes
#     inputs = tf.random.normal((10, 20, 30))
#     valid_lens = tf.constant(
#         [15, 18, 20, 17, 16, 19, 20, 14, 16, 13], dtype=inputs.dtype
#     )
#     masked_inputs = padding_mask_layer(inputs, query_len=4, valid_lens=valid_lens)
#     assert masked_inputs.shape == inputs.shape
