import os

import pytest
import tensorflow as tf
import numpy as np
from keras import regularizers

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
    seq_length = 10
    vocab_size = 32

    @staticmethod
    def create_toy_dataset(
        num_samples=1000, seq_length=10, vocab_size=64, num_classes=2
    ):
        # x = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
        x = np.random.normal(
            vocab_size / 2, vocab_size / 2 - 1, size=(num_samples, seq_length)
        )
        y = np.random.randint(0, 2, size=(num_samples, 1))
        y = np.random.normal(1, 1, size=(num_samples, seq_length))

        x_train = tf.random.uniform(
            shape=(num_samples, seq_length), maxval=vocab_size, dtype=tf.int32
        )
        y_train = tf.random.uniform(
            shape=(num_samples, 1), maxval=num_classes, dtype=tf.int32
        )
        return x_train, y_train

    @pytest.fixture
    def model(self):
        kernel_regularizer = regularizers.L1L2(l1=1e-5, l2=1e-4)
        inputs = tf.keras.layers.Input(shape=[self.seq_length])
        valid_lens = tf.keras.layers.Input(shape=())
        encoder = TransformerEncoder(
            vocab_size=self.vocab_size,
            maxlen_position_encoding=self.seq_length,
            d_model=64,
            num_heads=1,
            n_blocks=1,
        )
        learning_rate_schedule = lambda x: 1e-4 * x
        outputs, attn_weights = encoder(
            inputs,
            learning_rate_schedule=learning_rate_schedule,
            kernel_regularizer=kernel_regularizer,
        )
        # outputs = tf.keras.layers.Dense(10, activation="relu")(inputs)
        outputs = tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=2,
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_regularizer,
        )(outputs)
        pooled_output = tf.keras.layers.GlobalAveragePooling1D()(outputs)
        predictions = tf.keras.layers.Dense(
            1, activation="sigmoid", kernel_regularizer=kernel_regularizer
        )(pooled_output)
        model = tf.keras.Model(inputs=[inputs], outputs=predictions)
        optimizer = tf.keras.optimizers.Adam(1e-4)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        print(model.summary())
        return model

    def test_training(self, model):
        x_train, y_train = self.create_toy_dataset(
            vocab_size=self.vocab_size, seq_length=self.seq_length, num_samples=100
        )
        history = model.fit(
            x_train, y_train, epochs=50, batch_size=16, validation_split=0.2
        )
        # tf.keras.mixed_precision.set_global_policy("mixed_float16")
        assert (
            history.history["accuracy"][-1] > 0.5
        ), "Training accuracy should be greater than 0.5"
        loss, accuracy = model.evaluate(x_train, y_train)

        assert accuracy > 0.5, "Evaluation accuracy should be greater than 0.5"

        prediction = model.predict(x_train)
        assert (
            0 <= prediction[0][0] <= 1
        ), "Prediction should be a probability value between 0 and 1"
