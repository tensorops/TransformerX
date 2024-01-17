import tensorflow as tf

from transformerx.layers.masks import BaseMask

"""Future features for LookAheadMask:
- Add flexibility to mask only a window of future tokens rather than all. Can add a mask_window argument to specify size of lookahead window.
- Support masking along sequence dimension for either query or key. Can add mask_query and mask_key flags like in PaddingMask.
- Improve efficiency by using a sparse tensor representation for the mask instead of dense where possible. Could check if q_len and k_len are large and use sparse mask.
- Add options to cache the mask between calls if q_len and k_len are fixed. Can add cache argument.
"""
class LookAheadMask(BaseMask):
    def build_mask(self, q_len, k_len, *args, **kwargs):
        # Create a test mask manually using tf.linalg.band_part: this creates upper triangular boolean matrix which then
        # will be multiplied by the mask_value which is 10-9, and then we add this new mask matrix which its upper
        # triangle values are 10-9 with the scores (attention) matrix. Later when we pass this matrix to the softmax,
        # they will become 0 since they are -inf numbers.
        mask = (
                1
                - tf.linalg.LinearOperatorLowerTriangular(
            tf.ones((q_len, k_len)), -1, 0
        ).to_dense()
        )
        return mask

    # class LookAheadMask(tf.keras.layers.Layer):
    #     def __init__(self, **kwargs):
    #         super(LookAheadMask, self).__init__(**kwargs)
    #
    #     def build(self, input_shape):
    #         super(LookAheadMask, self).build(input_shape)
    #
    #     def call(self, inputs, **kwargs):
    #         sequence_length = tf.shape(inputs)[1]
    #
    #         # Create a lower triangular matrix with ones
    #         mask = tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
    #
    #         # Expand the mask to the batch dimension
    #         mask = tf.expand_dims(mask, 0)
    #         mask = tf.tile(mask, [tf.shape(inputs)[0], 1, 1])
    #
    #         return mask

    def compute_output_shape(self, input_shape):
        return input_shape
