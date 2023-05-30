import tensorflow as tf

from ..masks import BaseMask


class GlobalAttentionMask(BaseMask):
    def build_mask(self, inputs, q_length, k_length):
        mask = tf.ones((1, 1, q_length, k_length))
        return 1 - mask


if __name__ == "__main__()":
    print(3)
