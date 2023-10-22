import tensorflow as tf

from transformerx.layers.masks import BaseMask


# todo: implement atomic sparse attention masks here
# class GlobalAttentionMask(BaseMask):
#     def build_mask(self, scores):
#         input_shape = tf.shape(scores)
#         if input_shape.shape == 4:
#             print("input shape: ", input_shape)
#             q_length = input_shape[2]
#             k_length = input_shape[3]
#         elif input_shape.shape == 3:
#             q_length = input_shape[2]
#
#         mask = tf.ones((1, 1, q_length, q_length))
#         return 1 - mask


class DilatedAttentionMask(BaseMask):
    def __init__(self, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.dilation_rate = dilation_rate

    def build_mask(self, q_len, k_len, scores=None, *args, **kwargs):
        max_len = tf.maximum(q_len, k_len)

        mask = 1 - tf.linalg.band_part(
            tf.ones((max_len, max_len)), -1, self.dilation_rate
        )

        if self.multihead:
            mask = tf.expand_dims(mask, axis=0)

        return mask


if __name__ == "__main__":
    da = DilatedAttentionMask()
    input_tensor = tf.random.uniform((2, 6, 6))
    mask = da(input_tensor)

    print(mask)
