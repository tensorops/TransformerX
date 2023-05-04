import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, inputs, *args, **kwargs):
        mask = self.build_mask(tf.shape(inputs))
        return tf.multiply(inputs, mask)


class AttentionMask(BaseMask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        q_seq_len = input_shape[1]
        k_seq_len = input_shape[2]
        print("input_shape: ", input_shape[1])
        print(tf.linalg.band_part(tf.ones((q_seq_len, k_seq_len)), -1, 0))
        mask = 1 - tf.linalg.band_part(tf.ones((q_seq_len, k_seq_len)), -1, 0) * -1e9
        mask = tf.expand_dims(mask, axis=0)
        # mask = tf.expand_dims(mask, axis=2)
        # mask = tf.tile(mask, [input_shape[0], 1, 1])
        print(mask)
        # return tf.expand_dims(mask, axis=0)
        return mask


if __name__ == "__main__":
    from transformerx.layers import DotProductAttention

    input_tensor = tf.random.uniform((2, 3, 6))
    q_input_tensor = tf.random.uniform((2, 6, 6))
    attn_o, attn_w = DotProductAttention()(q_input_tensor, input_tensor, input_tensor)
    print("attn_o.shape: ", attn_o.shape)
    print("attn_w.shape:", attn_w.shape)
    print("attn_w:", attn_w)
    mask = AttentionMask()
    output_tensor = mask(attn_w)
    print(output_tensor)
    print(tf.nn.softmax(output_tensor, axis=-1))
