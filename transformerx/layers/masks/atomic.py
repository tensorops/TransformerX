import tensorflow as tf

from transformerx.layers.masks import BaseMask


class GlobalAttentionMask(BaseMask):
    def build_mask(self, inputs):
        input_shape = tf.shape(inputs)
        if input_shape.shape == 4:
            print("input shape: ", input_shape)
            q_length = input_shape[2]
            k_length = input_shape[3]
        elif input_shape.shape == 3:
            q_length = input_shape[2]

        mask = tf.ones((1, 1, q_length, q_length))
        return 1 - mask


if __name__ == "__main__":
    gm = GlobalAttentionMask()
    input_tensor = tf.random.uniform((2, 4, 6))
    mask = gm(input_tensor)

    print(mask)
