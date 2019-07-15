import numpy as np
import os
import preprocessing 
from w2v import train_word2vec

import tensorflow as tf
from tensorflow import keras

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 모델 타입. 김윤님의 Convolutional Neural Networks for Sentence Classification에 나와있는 대로 작성
# CNN-rand
# CNN-non-static
# CNN-static
model_type = "CNN-static"
embedding_mode = "pretrained_ko"

embedding_dim = 200
filter_sizes = (3, 4, 5)
num_filters = 100
dropout = 0.5
hidden_dims = 100

batch_size = 50
num_epochs = 10
min_word_count = 1
context = 10

def load_data():
    x, y, vocabulary, vocabulary_inv_list = preprocessing.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv

x_train, y_train, x_test, y_test, vocabulary_inv = load_data()
sequence_length = x_test.shape[1]

if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])

elif model_type == "CNN-rand":
    embedding_weights = None

if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = keras.layers.Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = keras.layers.Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = keras.layers.Dropout(dropout)(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = keras.layers.Conv1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = keras.layers.MaxPooling1D(pool_size=2)(conv)
    conv = keras.layers.Flatten()(conv)
    conv_blocks.append(conv)
z = keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = keras.layers.Dropout(dropout)(z)
z = keras.layers.Dense(hidden_dims, activation="relu")(z)
model_output = keras.layers.Dense(1, activation="sigmoid")(z)

model = keras.Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)
