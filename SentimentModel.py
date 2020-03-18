from ProjectUtils import GloveLoader
import pandas as pd
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import tensorflow as tf


MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
GLOVE_DIR = ''
dataset_path = 'Sentiment_Analysis_Dataset.csv'

df = pd.read_csv(dataset_path,
                 error_bad_lines=False,
                 encoding="ISO-8859-1",
                 dtype={"ItemID": int, "Sentiment": int, "SentimentSource": str, "SentimentText": str, },
                 warn_bad_lines=False, )

texts = df['SentimentText'].to_list()  # list of text samples
labels = df['Sentiment'].to_list()  # list of label ids
n_labels = max(labels)+1

print('Found %s texts.' % len(texts))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print('Saving tokenizer.')
pd.DataFrame({'data':[tokenizer]}).to_pickle('tokenizer.pickle')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# embedding_df = pd.read_csv(GLOVE_DIR+ 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt',delimiter=' ',header=None)
#
# embeddings_index = {word:embedding for word in embedding_df.iloc[:,0] for embedding in embedding_df.iloc[:,1:].to_numpy()}

# print('Found %s word vectors.' % len(embedding_df))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Embedding

print('preparing embedding layer')
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


def make_model():
    from keras.models import Input, Model
    from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
    from keras.optimizers import Adam

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(64, 5)(embedded_sequences)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = MaxPooling1D()(x)

    x = Conv1D(128, 5)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    # x = MaxPooling1D(5)(x)
    # x = Conv1D(256, 5)(x)
    # x = LeakyReLU()(x)
    # x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)

    x = MaxPooling1D()(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(50)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    # x = Dense(256)(x)
    # x = LeakyReLU()(x)
    # x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)

    preds = Dense(n_labels, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    model.summary()
    return model


def make_sentiment_model():
    from keras.layers import Embedding, Bidirectional, Dropout, Dense, LSTM
    from keras.models import Sequential

    activation_fn = 'sigmoid'
    dropout_factor = 0.2

    model = Sequential([
        Embedding(len(word_index) + 1,
                  EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=False),

        Bidirectional(LSTM(100,
                           return_sequences=True,
                           activation=activation_fn,
                           kernel_initializer='normal')),
        Dropout(dropout_factor),
        Bidirectional(LSTM(100 * 2,
                           return_sequences=True,
                           activation=activation_fn,
                           kernel_initializer='normal')),
        Dropout(dropout_factor),
        Bidirectional(LSTM(100,
                           activation=activation_fn,
                           kernel_initializer='normal')),
        Dropout(dropout_factor),
        Dense(100, activation=activation_fn, kernel_initializer='normal'),
        Dropout(dropout_factor),
        Dense(60, activation=activation_fn, kernel_initializer='normal'),
        Dropout(dropout_factor),
        Dense(2, activation='softmax', kernel_initializer='normal')
    ])
    print(model.summary())

    print("Compiling model.")
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model


# create model
model = make_model()

# Load old model for further training if it exists
if os.path.exists('sentiment_model.h5'):
    print('Loading old model.')
    from keras.models import load_model

    model = load_model("sentiment_model.h5")  # loaded bot object

# train
print("Training model.")

# fit the model
model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=32)

# save model
print('Saving model.')
model.save("sentiment_model.h5")

# evaluate the model
print('Testing model.')
loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print('Accuracy: %f' % (accuracy * 100))
