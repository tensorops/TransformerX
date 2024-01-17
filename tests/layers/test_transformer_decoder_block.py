import pytest
import tensorflow as tf

from transformerx.layers import TransformerDecoderBlock


def test_transformer_decoder_block():
    batch_size = 2
    seq_length = 10
    queries = tf.random.uniform((batch_size, seq_length, 64))
    keys = tf.random.uniform((batch_size, seq_length, 64))
    values = tf.random.uniform((batch_size, seq_length, 64))
    valid_lens = tf.ones((batch_size, seq_length))

    decoder_block = TransformerDecoderBlock(d_model=64)
    output, attn1_weights, attn2_weights = decoder_block(
        queries, keys, values, valid_lens
    )

    assert output.shape == (batch_size, seq_length, 64)
    assert attn1_weights.shape == (batch_size, 8, seq_length, seq_length)
    assert attn2_weights.shape == (batch_size, 8, seq_length, seq_length)


class TestTransformerDecoderBlock:
    def test_basic_functionality(self):
        batch_size = 2
        seq_length = 10
        queries = tf.random.uniform((batch_size, seq_length, 512))
        keys = tf.random.uniform((batch_size, seq_length, 512))
        values = tf.random.uniform((batch_size, seq_length, 512))
        valid_lens = tf.ones((batch_size, seq_length))

        decoder_block = TransformerDecoderBlock()
        output, attn1_weights, attn2_weights = decoder_block(
            queries, keys, values, valid_lens
        )

        assert output.shape == (batch_size, seq_length, 512)
        assert attn1_weights.shape == (batch_size, 8, seq_length, seq_length)
        assert attn2_weights.shape == (batch_size, 8, seq_length, seq_length)

    @pytest.fixture
    def transformer_block(self):
        return TransformerDecoderBlock()

    def test_call(self, transformer_block):
        # Generate inputs for testing
        queries = tf.ones(shape=(2, 5, 512))
        keys = tf.ones(shape=(2, 10, 512))
        values = tf.ones(shape=(2, 10, 512))
        valid_lens = tf.constant([5, 7])

        # Test the call method
        output, _, _ = transformer_block(queries, keys, values, valid_lens)
        assert output.shape == (2, 5, 512)

    def test_call2(self):
        batch_size = 2
        seq_length = 5
        hidden_size = 4
        num_heads = 2
        dropout_rate = 0.2
        epsilon = 1e-6

        decoder_block = TransformerDecoderBlock(hidden_size, num_heads, dropout_rate)

        input_tensor = tf.random.uniform(
            shape=(batch_size, seq_length, hidden_size), dtype=tf.float32
        )
        look_ahead_mask = 1 - tf.linalg.band_part(
            tf.ones((seq_length, seq_length)), -1, 0
        )
        look_ahead_mask = tf.tile(
            tf.expand_dims(look_ahead_mask, 0), [batch_size, 1, 1]
        )

        output_tensor, _, attention_weights = decoder_block(
            input_tensor, input_tensor, input_tensor, look_ahead_mask
        )

        # Check the shape of output_tensor
        assert output_tensor.shape == (batch_size, seq_length, hidden_size)

        # Check the shape of attention_weights
        assert attention_weights.shape == (
            batch_size,
            num_heads,
            seq_length,
            seq_length,
        )

        # Check that the output tensor is not all zeros
        assert not tf.reduce_all(tf.abs(output_tensor) < epsilon)

        # Check that the attention weights sum to 1 along the last axis
        assert tf.reduce_all(
            tf.abs(tf.reduce_sum(attention_weights, axis=-1) - 1) < epsilon
        )

        tf.debugging.assert_near(
            tf.reduce_sum(attention_weights, axis=-1),
            tf.ones((batch_size, num_heads, seq_length)),
            atol=epsilon,
        )

    def test_shape(self, transformer_block):
        # Test that the output shape is correct
        queries = tf.ones(shape=(2, 5, 512))
        keys = tf.ones(shape=(2, 10, 512))
        values = tf.ones(shape=(2, 10, 512))
        valid_lens = tf.constant([5, 7])
        output, _, _ = transformer_block(queries, keys, values, valid_lens)
        assert output.shape == (2, 5, 512)

    def test_addnorm(self, transformer_block):
        # Test that the output of the addnorm layer is correct
        queries = tf.ones(shape=(2, 5, 512))
        keys = tf.ones(shape=(2, 10, 512))
        values = tf.ones(shape=(2, 10, 512))
        valid_lens = tf.constant([5, 7])
        output, _, _ = transformer_block(queries, keys, values, valid_lens)
        assert transformer_block.addnorm1(output, queries).shape == (2, 5, 512)

    def test_multihead_attention(self, transformer_block):
        # Test that the output of the multi-head attention layer is correct
        queries = tf.ones(shape=(2, 5, 512))
        keys = tf.ones(shape=(2, 10, 512))
        values = tf.ones(shape=(2, 10, 512))
        valid_lens = tf.constant([5, 7])
        output, attn_weights = transformer_block.attention1(queries, queries, queries)
        assert output.shape == (2, 5, 512)  # (batch_size, sequence_length, d_model
        assert attn_weights.shape == (2, 8, 5, 5)  # (batch_size, num_heads, sequence_length, sequence_length)

    def test_positionwise_ffn(self, transformer_block):
        # Test that the output of the position-wise feedforward network is correct
        queries = tf.ones(shape=(2, 5, 512))
        keys = tf.ones(shape=(2, 10, 512))
        values = tf.ones(shape=(2, 10, 512))
        valid_lens = tf.constant([5, 7])
        output = transformer_block.ffn(queries)
        assert output.shape == (2, 5, 512)

    def test_clip_norm(self, transformer_block):
        block = TransformerDecoderBlock(
            d_model=512, num_heads=8, input_hidden_units_ffn=2048
        )
        block.clip_norm = 1.0
        queries = tf.ones(shape=(2, 5, 512))
        keys = tf.ones(shape=(2, 10, 512))
        values = tf.ones(shape=(2, 10, 512))
        valid_lens = tf.constant([5, 7])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([queries, keys, values])
            output, _, attn_weights = block(
                queries, keys, values, attention_mask=valid_lens
            )
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, [queries, keys, values])
        clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=block.clip_norm)
        # Check that the norm of the clipped gradients is less than or equal to clip_norm
        for clipped_grad in clipped_grads:
            assert tf.math.reduce_euclidean_norm(clipped_grad) <= block.clip_norm

    def htest_mixed_precision(self, transformer_block):
        # Test that the layer can run with mixed precision
        # tf.config.experimental.set_memory_growth(
        #     tf.config.list_physical_devices("GPU")[0], True
        # )
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # Create test inputs and labels
        input_tensor = tf.ones(shape=(2, 5, 512), dtype=tf.float16)
        output_tensor, _, _ = transformer_block(
            input_tensor, input_tensor, input_tensor
        )

        # Check that the output tensor is of the correct shape and dtype
        assert output_tensor.shape == (2, 5, 512)
        assert output_tensor.dtype == tf.float16
        # tf.keras.mixed_precision.set_global_policy("float32")
