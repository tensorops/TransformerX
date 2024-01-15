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

        mask = tf.ones((max_len, max_len), dtype=tf.float32)

        #todo: correct this. it must return the correct values for the dilated attention mask.

        mask_bool = tf.math.logical_and(
            tf.math.abs(tf.range(max_len) - tf.range(max_len)[:, tf.newaxis])
            <= self.dilation_rate,
            tf.math.not_equal(tf.range(max_len)[:, tf.newaxis], tf.range(max_len)),
        )

        # Convert the boolean mask to float32
        mask_float = tf.cast(mask_bool, dtype=tf.float32)

        # Multiply the original mask with the dilated mask to apply the dilation
        dilated_mask = mask * mask_float

        if self.multihead:
            mask = tf.expand_dims(mask, axis=0)


        return mask


if __name__ == "__main__":
    da = DilatedAttentionMask()
    input_tensor = tf.random.uniform((2, 6, 6))
    mask = da(input_tensor)

    print(mask)
