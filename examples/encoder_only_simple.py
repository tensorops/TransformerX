import tensorflow as tf
from transformerx.layers.transformer_encoder import TransformerEncoder
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# A tiny model TransformerEncoder
def encoder_only_model(vocab_size, d_model, num_heads, n_blocks, max_seq_length, dropout_rate=0.1):
    inputs = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32)

    # Initialize and apply the TransformerEncoder
    transformer_encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        n_blocks=n_blocks,
        maxlen_position_encoding=max_seq_length,
        dropout_rate=dropout_rate,
    )

    # Apply the transformer encoder to the input sequence
    encoder_output, _ = transformer_encoder(inputs)

    # Global average pooling to obtain a fixed-size representation
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(encoder_output)

    # Dense layer for classification (modify as needed for your specific task)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(pooled_output)

    # Build and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # Load IMDb dataset
    # A collection of 25,000 movie reviews sourced from IMDB, categorized based on sentiment (positive/negative).
    # The reviews have undergone preprocessing, and each review is represented as a list of word indexes (integers).

    # Customize the following hyperparameters
    vocab_size = 1000  # Replace with the actual vocabulary size.
    d_model = 8
    num_heads = 4
    n_blocks = 2
    max_seq_length = 100
    dropout_rate = 0.1

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=max_seq_length)
    x_train = tf.convert_to_tensor(x_train, dtype=None, dtype_hint=None, name=None)

    x_test = pad_sequences(x_test, maxlen=max_seq_length)
    x_test = tf.convert_to_tensor(x_test, dtype=None, dtype_hint=None, name=None)

    y_train = tf.convert_to_tensor(y_train[:], dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test[:], dtype=tf.int32)

    # Building the model

    bert_like_model = encoder_only_model(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        n_blocks=n_blocks,
        max_seq_length=max_seq_length,
        dropout_rate=dropout_rate,
    )
    # Model summary
    print(bert_like_model.summary())
    bert_like_model.fit(x_train.numpy(), y_train, epochs=5, batch_size=32)

    # Evaluating the model on the test set
    test_loss, test_accuracy = bert_like_model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
