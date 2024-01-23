import tensorflow as tf
from keras.losses import MeanSquaredError

from transformerx.layers import (
    TransformerEncoderBlock,
    MultiHeadAttention,
    AddNorm,
    PositionwiseFFN,
)
from transformerx.layers.transformer_encoder import TransformerEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Load IMDb dataset
max_features = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
x_train = tf.convert_to_tensor(x_train, dtype=None, dtype_hint=None, name=None)
y_train = tf.convert_to_tensor(y_train[:, 0], dtype=tf.int32)

print("x train shape imdb: ", x_train[0])
print("y train shape imdb: ", y_train[0])


def bert_encoder_model(
    vocab_size, d_model, num_heads, n_blocks, max_seq_length, dropout_rate=0.1
):
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # Example usage
    vocab_size = 10000  # Replace with the actual vocabulary size
    d_model = 8
    num_heads = 1
    n_blocks = 1
    max_seq_length = 200
    dropout_rate = 0.1

    bert_model = bert_encoder_model(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        n_blocks=n_blocks,
        max_seq_length=max_seq_length,
        dropout_rate=dropout_rate,
    )
    # dense_model_instance = dense_model(vocab_size=vocab_size, max_seq_length=max_seq_length)

    # Display the model summary
    print(bert_model.summary())
    # print(dense_model_instance.summary())

    # Train the model with your data (replace with actual data)
    # x_train = tf.random.uniform(
    #     shape=(1000, max_seq_length), maxval=vocab_size, dtype=tf.int32
    # )
    # print("x train shape random: ", x_train[0])
    # y_train = tf.random.uniform(shape=(1000,), maxval=2, dtype=tf.int32)
    # dense_model_instance.fit(x_train, y_train, epochs=30, batch_size=32)
    bert_model.fit(x_train.numpy(), y_train, epochs=60, batch_size=32)


if __name__ == "__main__":
    main()


# import tensorflow as tf


# Reshape the input tensors to include a sequence length dimension
# seq_length = x_train.shape[1]
# x_train_reshaped = x_train.reshape((-1, seq_length, 1))
# x_test_reshaped = x_test.reshape((-1, seq_length, 1))

# Define a tiny model with TransformerEncoderBlock layer
# Define a tiny model with TransformerEncoderBlock layer using the functional API
# def create_model():
#     inputs = layers.Input(shape=(maxlen,))
#     embedding = layers.Embedding(max_features, 8)(inputs)
#
#     # Transformer Encoder Block as a layer
#     transformer_encoder, _ = TransformerEncoderBlock(d_model=8, num_heads=4, dropout_rate=0.1)(embedding)
#
#     # Flatten and Dense layers for classification
#     flatten = layers.Flatten()(transformer_encoder)
#     outputs = layers.Dense(2, activation='softmax')(flatten)
#
#     return models.Model(inputs, outputs)
#
# # Instantiate the model
# model = create_model()
#
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Display the model summary
# model.summary()
#
# # Train the model
# history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
#
# # Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f'Test accuracy: {test_acc}')
