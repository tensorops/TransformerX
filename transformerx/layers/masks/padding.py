import tensorflow as tf

from transformerx.layers.masks import BaseMask

"""Future features for PaddingMask:
- Make the padding mask creation more flexible by supporting different padding directions (left, right, both). Can add 
a padding_direction argument to allow masking left, right or both sides.
- Support masking along the sequence dimension for either the query or key by adding mask_query and mask_key boolean 
flags.
- Allow masking a subset of heads in the multi-head attention by adding a heads argument that specifies which heads to 
mask.
- Use a sparse tensor for the mask instead of a dense tensor when possible, to improve memory and compute efficiency. 
The use_sparse_tensor_mask flag can control this.
- Add options to automatically pad the query/key to the maximum sequence length to avoid having to pass valid lengths. 
Can add auto_pad argument.
- Improve performance by avoiding unnecessary casts and expansions when applying the mask.
"""


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
            input_shape=None,
            *args,
            **kwargs,
    ):
        """Build padding mask.

        This method expects a padded input sequence such that they all have the same length.

        Parameters
        ----------
        q_len
        k_len
        valid_lens
        padding_mask
        scores
        input_shape
        args
        kwargs

        Returns
        -------

        """

        if padding_mask is not None:
            mask = tf.cast(
                padding_mask,
                dtype=self.scores_dtype,
            )

            # mask = tf.cast(padding_mask, dtype=self.scores_dtype)

        # fixme: later these functionality will be added again.
        #  problem: it should construct a padding mask based on the valid_lens(valid lengths) tensor where each
        #  sequence in the batch should be padded wrt the valid_lens tensor such that the final mask has the same
        #  shape as the scores matrix so they can be added up together in the `call()` method in the BaseMask class.

        # elif valid_lens is not None:
        #     max_len = tf.maximum(q_len, k_len)
        #     # max_len = tf.reduce_max(valid_lens)
        #     print("here", self.padding_value)
        #     mask = 1 - tf.sequence_mask(valid_lens, max_len, dtype=self.scores_dtype)
        #     if input_shape:
        #         if len(input_shape) == 3:
        #             mask = tf.reshape(padding_mask, (input_shape[0], 1, input_shape[-1]))
        #         elif len(input_shape) == 4:
        #             mask = tf.reshape(padding_mask, (input_shape[0], 1, 1, input_shape[-1]))
        #     else:
        #         raise Exception("Input shape must be provided to reconstruct the mask with the same shape")
        #     print("mask: ", mask)

        # receives the scores matrix and derive the padding mask before passing to the softmax
        elif scores is not None:
            mask = tf.cast(
                tf.math.equal(scores, self.padding_value), dtype=scores.dtype
            )
        else:
            raise ValueError(
                "Either 'valid_lens', 'padding_mask' or \"'scores' along with the padding_value\" must be "
                "provided."
            )
        print("mask : ", mask)
        return mask


if __name__ == "__main__":
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

    # Test with padding values
    scores = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=tf.float32)

    padding_mask3 = PaddingMask(padding_value=1)
    masked = padding_mask3(scores)

    expected = tf.constant([[-1e9, 2, 3, 0], [4, 5, 0, 0]])
    print("here is the padding 1: ", masked)

    assert tf.reduce_all(tf.equal(masked, expected))

    # Test with multi-head
    scores = tf.constant(
        [[[1, 2, 3, 0], [4, 5, 0, 0]], [[1, 2, 0, 0], [4, 5, 6, 0]]], dtype=tf.float32
    )

    print("All tests passed!")

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
