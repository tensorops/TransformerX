import pytest
import tensorflow as tf
import numpy as np

from transformerx.layers import TransformerEncoder


class TestTransformerEncoder:
    @pytest.fixture(scope="class")
    def encoder(self):
        return TransformerEncoder(
            vocab_size=1000,
            maxlen_position_encoding=50,
            d_model=128,
            num_heads=4,
            n_blocks=2,
        )

    def test_embedding_output_shape(self, encoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        embedded_data = encoder.embedding(input_data)
        assert embedded_data.shape == (2, 3, 128)

    def test_positional_encoding_output_shape(self, encoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        embedded_data = encoder.embedding(input_data)
        pos_encoded_data = encoder.pos_encoding(embedded_data)
        assert pos_encoded_data.shape == (2, 3, 128)

    def test_encoder_block_output_shape(self, encoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        valid_lens = tf.constant([3, 2], dtype=tf.float32)
        embedded_data = encoder.embedding(input_data)
        pos_encoded_data = encoder.pos_encoding(embedded_data)
        block_output, block_attn_weights = encoder.blocks[0](pos_encoded_data)
        assert block_output.shape == (2, 3, 128)

    def test_encoder_output_shape(self, encoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        valid_lens = tf.constant([3, 2], dtype=tf.float32)
        output, attn_weights = encoder(input_data)
        assert output.shape == (2, 3, 128)

    def test_encoder_output_values(self, encoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        valid_lens = tf.constant([3, 2], dtype=tf.float32)
        output, attn_weights = encoder(input_data)
        assert not np.allclose(output.numpy(), np.zeros((2, 3, 128)))

    def test_encoder_attention_weights_shape(self, encoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        valid_lens = tf.constant([3, 2], dtype=tf.float32)
        _ = encoder(input_data)
        for attention_weights in encoder.attention_weights:
            assert attention_weights.shape == (2, 4, 3, 3)

    def test_encoder_attention_weights_values(self, encoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        valid_lens = tf.constant([3, 2], dtype=tf.float32)
        _ = encoder(input_data)
        for attention_weights in encoder.attention_weights:
            assert not np.allclose(attention_weights.numpy(), np.zeros((2, 4, 3, 3)))


class TestTransformerEncoderIntegration:
    @staticmethod
    def create_toy_dataset(
        num_samples=1000, seq_length=10, vocab_size=64, num_classes=2
    ):
        x = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
        y = np.random.randint(0, 2, size=(num_samples, 1))

        x_train = tf.random.uniform(
            shape=(num_samples, seq_length), maxval=vocab_size, dtype=tf.int32
        )
        y_train = tf.random.uniform(
            shape=(num_samples, 1), maxval=num_classes, dtype=tf.int32
        )
        return x_train, y_train

    @pytest.fixture
    def model(self):
        seq_length = 10
        vocab_size = 64
        inputs = tf.keras.layers.Input(shape=[seq_length])
        valid_lens = tf.keras.layers.Input(shape=())
        encoder = TransformerEncoder(
            vocab_size=vocab_size, maxlen_position_encoding=seq_length
        )
        outputs, attn_weights = encoder(inputs)
        pooled_output = tf.keras.layers.GlobalAveragePooling1D()(outputs)
        predictions = tf.keras.layers.Dense(1, activation="sigmoid")(pooled_output)
        model = tf.keras.Model(inputs=[inputs], outputs=predictions)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def test_training(self, model):
        x_train, y_train = self.create_toy_dataset()
        history = model.fit(
            x_train, y_train, epochs=5, batch_size=32, validation_split=0.2
        )
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        assert (
            history.history["accuracy"][-1] > 0.5
        ), "Training accuracy should be greater than 0.5"
        loss, accuracy = model.evaluate(x_train, y_train)

        assert accuracy > 0.5, "Evaluation accuracy should be greater than 0.5"

        prediction = model.predict(x_train)
        assert (
            0 <= prediction[0][0] <= 1
        ), "Prediction should be a probability value between 0 and 1"
