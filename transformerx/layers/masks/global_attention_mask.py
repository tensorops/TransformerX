import tensorflow as tf


class GlobalAttentionMask:
    def __init__(self, mask_type="none", mask_prob=0.0, dilation_rate=1):
        self.mask_type = mask_type
        self.mask_prob = mask_prob
        self.dilation_rate = dilation_rate

    def get_mask(self, input_shape):
        if len(input_shape) == 4:
            # Assumes the input shape is 4-d ('b', 'h', 'l', 'd')
            input_shape = input_shape[1:]
            batch_size, seq_len = input_shape[0], input_shape[2]
        elif len(input_shape) == 3:
            # Assumes the input shape is 3-d ('b', 'l', 'd')
            batch_size, seq_len = input_shape[0], input_shape[1]
        elif len(input_shape) == 2:
            # Assumes the input shape is 2-d ('b', 'd')
            batch_size, seq_len = input_shape[0], input_shape[1]
        else:
            raise ValueError(
                "The input shape must be 2-d ('b', 'd'), 3-d ('b', 'l', 'd') or 4-d ('b', 'h', 'l', 'd')"
            )

        mask = tf.ones((batch_size, seq_len, seq_len), dtype=tf.float32)

        if self.mask_type == "none":
            pass

        elif self.mask_type == "random":
            mask = tf.where(
                tf.random.uniform((batch_size, seq_len, seq_len)) < self.mask_prob,
                tf.zeros((batch_size, seq_len, seq_len)),
                mask,
            )
        elif self.mask_type == "dilated":
            mask = self.create_dilated_mask(mask, self.dilation_rate)

        return mask

    # create a dilated mask method
    def create_dilated_mask(self, mask, dilation_rate):
        batch_size, seq_len = mask.shape[0], mask.shape[1]

        # Create a boolean mask where True indicates positions that need to be masked
        mask_bool = tf.math.logical_and(
            tf.math.abs(tf.range(seq_len) - tf.range(seq_len)[:, tf.newaxis])
            <= dilation_rate,
            tf.math.not_equal(tf.range(seq_len)[:, tf.newaxis], tf.range(seq_len)),
        )

        # Convert the boolean mask to float32
        mask_float = tf.cast(mask_bool, dtype=tf.float32)

        # Multiply the original mask with the dilated mask to apply the dilation
        dilated_mask = mask * mask_float
        return dilated_mask
