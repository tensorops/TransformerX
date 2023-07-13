import tensorflow as tf

from transformerx.layers.core import BaseMask


class LookAheadMask(BaseMask):
    def build_mask(self, q_len, k_len, *args, **kwargs):
        mask = (
            1
            - tf.linalg.LinearOperatorLowerTriangular(
                tf.ones((q_len, k_len)), -1, 0
            ).to_dense()
        )
        return mask


class PaddingMask(BaseMask):
    def __init__(self, padding_value=1, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def build_mask(
        self,
        q_len,
        k_len,
        valid_lens=None,
        padding_mask=None,
        scores=None,
        *args,
        **kwargs,
    ):
        if padding_mask is not None:
            mask = tf.cast(
                tf.math.logical_not(tf.cast(padding_mask, dtype=tf.bool)),
                dtype=self.scores_dtype,
            )
        elif valid_lens is not None:
            max_len = tf.maximum(q_len, k_len)
            mask = tf.sequence_mask(valid_lens, max_len, dtype=self.scores_dtype)
            print("here", self.padding_value)
            if self.padding_value == 1:
                mask = 1 - mask
            else:
                mask = tf.where(
                    tf.cast(mask, dtype=tf.bool), self.padding_value, scores
                )
                # print(
                #     tf.where(mask, self.padding_value, scores),
                #     tf.where(mask, self.padding_value),
                # )

        elif scores is not None:
            mask = tf.cast(
                tf.math.equal(scores, self.padding_value), dtype=scores.dtype
            )
        else:
            raise ValueError("Either 'valid_lens' or 'padding_mask' must be provided.")

        if self.multihead:
            mask = tf.expand_dims(mask, axis=1)
        # else:
        #     mask = tf.expand_dims(1 - mask, axis=1)
        print("mask : ", mask)
        return mask


if __name__ == "__main__":
    # from transformerx.layers import DotProductAttention, MultiHeadAttention
    #
    # input_tensor = tf.random.uniform((2, 4, 6))
    # q_input_tensor = tf.random.uniform((2, 4, 6))
    # attn_o, attn_w = DotProductAttention()(q_input_tensor, q_input_tensor, input_tensor)
    #
    # print("attn_w.shape: ", attn_w.shape)
    # la_mask = LookAheadMask()
    # output_tensor = la_mask(attn_w)
    # print("output tensor and its shape: ", output_tensor.shape, output_tensor)
    #
    # multihead_attn = MultiHeadAttention(d_model=32, num_heads=4, dropout_rate=0.1)
    # output, attn_w = multihead_attn(q_input_tensor, input_tensor, input_tensor)
    #
    # sample_input = tf.random.uniform((1, 1, 4, 2))
    # output_tensor = la_mask(attn_w[0])
    # print("output_tensor.shape la_mask: ", output_tensor.shape, output_tensor)
    # output_tensor = la_mask(sample_input)
    # print("output_tensor.shape la_mask2: ", output_tensor.shape, output_tensor)
    #
    # data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    # # Create a 2D tensor
    # data = tf.constant([[1, 2, 3], [4, 5, 6]])
    #
    # # Convert the dataset to a tensor
    # # data_tensor = tf.constant(data, dtype=tf.float32)
    #
    # # Create a SequencePadding layer
    # # sequence_padding_layer = PaddingLayer(0, 4)
    #
    # # padded_data = sequence_padding_layer(data)
    #
    # # Test input
    # # input_tensor = tf.constant(
    # #     [
    # #         [[1, 2, 0], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
    # #         [[1, 2, 3], [4, 5, 0], [0, 0, 0], [0, 0, 0]],
    # #     ],
    # #     dtype=tf.float32,
    # # )
    #
    # inputs = tf.random.normal(
    #     (2, 3, 4)
    # )  # 3D input tensor (batch_size=2, query_len=3, key_dim=4)
    # valid_lens = tf.constant(
    #     [2, 3], dtype=tf.int32
    # )  # Valid lengths for each sequence in the batch
    #
    # # Create a PaddingMask layer
    # padding_mask_layer = PaddingMask(multihead=False)
    #
    # masked_inputs = padding_mask_layer(inputs, query_len=3, valid_lens=valid_lens)
    #
    # print("Inputs:")
    # print(inputs)
    # print("\nValid Lens Masked Inputs:")
    # print(masked_inputs)
    # # Generate the padding mask
    # # padding_mask = padding_mask_layer(input_tensor)
    # # print(padding_mask.shape, padding_mask)
    #
    # lad_mask = la_mask(input_tensor)
    # # print(lad_mask.shape, lad_mask)
    #
    # # Test with `padding_mask` option
    # padding_mask = tf.constant(
    #     [[False, False, True, True], [False, False, False, True]]
    # )  # Example padding mask
    # masked_inputs = padding_mask_layer(
    #     inputs, padding_mask=padding_mask, query_len=3, key_len=4
    # )
    # print("\nPadding Masked Inputs:")
    # print(masked_inputs)
    #
    # # Test with `inputs` option
    # # Test with 3D tensor
    # inputs_3d = tf.random.normal(
    #     (2, 3, 4)
    # )  # 3D input tensor (batch_size=2, query_len=3, key_dim=4)
    # masked_inputs_3d = padding_mask_layer(inputs_3d)
    # print("\n3D Inputs Masked Inputs:")
    # print(masked_inputs_3d)
    #
    # # Test with 4D tensor
    # inputs_4d = tf.random.normal(
    #     (2, 3, 4, 5)
    # )  # 4D input tensor (batch_size=2, heads=3, q_dim=4, k_dim=5)
    # masked_inputs_4d = padding_mask_layer(scores=inputs_4d)
    # print("\n4D Inputs Masked Inputs:")
    # print(masked_inputs_4d)

    # Test with valid lengths
    scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)
    valid_lens = tf.constant([3, 2])

    padding_mask = PaddingMask()
    masked = padding_mask(scores, valid_lens=valid_lens)

    expected = tf.constant([[1, 2, 3, -1e9], [4, 5, -1e9, -1e9]])

    assert tf.reduce_all(tf.equal(masked, expected))

    # Test with padding values
    scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)

    padding_mask2 = PaddingMask(padding_value=0)
    masked = padding_mask2(scores)

    expected = tf.constant([[1, 2, 3, -1e9], [4, 5, -1e9, -1e9]])

    assert tf.reduce_all(tf.equal(masked, expected))

    # Test with multi-head
    scores = tf.constant(
        [[[1, 2, 3, 0], [4, 5, 0, 0]], [[1, 2, 0, 0], [4, 5, 6, 0]]], dtype=tf.float32
    )

    padding_mask = PaddingMask(multihead=True)
    masked = padding_mask(scores, valid_lens=tf.constant([3, 2]))

    expected = tf.constant(
        [[[1, 2, 3, -1e9], [4, 5, 0, -1e9]], [[1, 2, -1e9, -1e9], [4, 5, -1e9, -1e9]]]
    )

    print(masked)
    assert tf.reduce_all(tf.equal(masked, expected))

    print("All tests passed!")
