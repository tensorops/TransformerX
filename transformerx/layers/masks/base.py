import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, inputs, *args, **kwargs):
        if len(inputs.shape) == 3:
            m_inputs = tf.expand_dims(inputs, axis=1)
        else:
            m_inputs = inputs
        mask = self.build_mask(tf.shape(m_inputs))
        return tf.multiply(inputs, mask)


class LookAheadMask(BaseMask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        q_seq_len = input_shape[2]
        k_seq_len = input_shape[3]
        mask = 1 - tf.linalg.band_part(tf.ones((q_seq_len, k_seq_len)), -1, 0)
        mask = tf.expand_dims(mask, axis=0)
        return mask


if __name__ == "__main__":
    from transformerx.layers import DotProductAttention, MultiHeadAttention

    input_tensor = tf.random.uniform((2, 4, 6))
    q_input_tensor = tf.random.uniform((2, 4, 6))
    attn_o, attn_w = DotProductAttention()(q_input_tensor, input_tensor, input_tensor)
    # print("mask attn_o.shape: ", attn_o.shape)
    # print("mask attn_w.shape:", attn_w.shape)
    # print("mask attn_w:", attn_w)
    mask = LookAheadMask()
    output_tensor = mask(attn_w)
    # print("masked ouptut shape: ", output_tensor.shape, output_tensor)
    # print(tf.nn.softmax(output_tensor, axis=-1))
    print(tf.nn.softmax(output_tensor, axis=-1))

    multihead_attn = MultiHeadAttention(d_model=32, num_heads=4, dropout_rate=0.1)
    output, attn_w = multihead_attn(q_input_tensor, input_tensor, input_tensor)
    output_tensor = mask(attn_w)
    # print("mask output_tensor.shape: ", output_tensor.shape)
    # print("mask output_tensor.shape: ", attn_w)
    # print(tf.nn.softmax(output_tensor, axis=-1))
