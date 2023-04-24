import tensorflow as tf
import pytest
from transformerx.layers import TransformerEncoderBlock


class TestTransformerEncoderBlock:
    @pytest.fixture
    def transformer_encoder_block(self):
        return TransformerEncoderBlock()

    def test_transformer_encoder_block_output_shape(self, transformer_encoder_block):
        x = tf.random.uniform((32, 10, 512))
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_attention_mask(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        attention_mask = tf.ones((32, 10))
        output_tensor, attn_weights = transformer_encoder_block(
            x, x, x, attention_mask=attention_mask
        )
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_residual_connections(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.residual_connections = (True, True)
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_clip_norm(self, transformer_encoder_block):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.clip_norm = 1.0
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert tf.math.reduce_max(tf.norm(output_tensor, axis=-1)) <= 1.0

    def test_transformer_encoder_block_with_layer_norm(self, transformer_encoder_block):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.use_norm = True
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_without_layer_norm(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.use_norm = False
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_bias(self, transformer_encoder_block):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.bias = True
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_without_bias(self, transformer_encoder_block):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.bias = False
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_mixed_precision(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        tf.keras.mixed_precision.set_global_policy("float32")
        transformer_encoder_block.mixed_precision = True
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.dtype == tf.float32

    def test_transformer_encoder_block_with_learning_rate_schedule(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.learning_rate_schedule = lambda x: 1e-4 * x
        output_tensor, attn_weights = transformer_encoder_block(x, x, x, global_step=10)
        assert transformer_encoder_block.learning_rate == 1e-3

    def test_transformer_encoder_block_with_kernel_regularizer(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_bias_regularizer(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.bias_regularizer = tf.keras.regularizers.l2(1e-4)
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_non_linear_proj(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.non_linear_proj = tf.keras.layers.Dense(256)
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_contextualized_embeddings(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        transformer_encoder_block.contextualized_embeddings = tf.keras.layers.Dense(768)
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)


class TestTransformerEncoderBlockIntegration:
    @pytest.fixture
    def transformer_encoder_block(self):
        return TransformerEncoderBlock()

    def test_transformer_encoder_block_with_embedding_layer(
        self, transformer_encoder_block
    ):
        input_data = tf.random.uniform((32, 10), minval=0, maxval=100, dtype=tf.int32)
        embedding_layer = tf.keras.layers.Embedding(input_dim=100, output_dim=512)
        x = embedding_layer(input_data)
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        assert output_tensor.shape == (32, 10, 512)

    def test_transformer_encoder_block_with_dense_layer(
        self, transformer_encoder_block
    ):
        x = tf.random.uniform((32, 10, 512))
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        dense_layer = tf.keras.layers.Dense(units=256, activation="relu")
        output_tensor = dense_layer(output_tensor)
        assert output_tensor.shape == (32, 10, 256)

    def test_transformer_encoder_block_training(self, transformer_encoder_block):
        input_data = tf.random.uniform((32, 10), minval=0, maxval=100, dtype=tf.int32)
        target_data = tf.random.uniform((32, 10), minval=0, maxval=100, dtype=tf.int32)

        input_layer = tf.keras.layers.Input(shape=(10,), dtype=tf.int32)
        embedding_layer = tf.keras.layers.Embedding(input_dim=100, output_dim=512)
        x = embedding_layer(input_layer)
        output_tensor, attn_weights = transformer_encoder_block(x, x, x)
        dense_layer = tf.keras.layers.Dense(units=100)
        output_tensor = dense_layer(output_tensor)

        model = tf.keras.Model(inputs=input_layer, outputs=output_tensor)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()

        with tf.GradientTape() as tape:
            predictions = model(input_data)
            loss = loss_object(target_data, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        assert loss is not None
        assert gradients is not None
