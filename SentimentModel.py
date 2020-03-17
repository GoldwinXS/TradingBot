from ProjectUtils import *
import pandas as pd
import os
import tensorflow as tf

""" Project Setup """
model = tf.keras.models.load_model("twitter_model_v3.h5")  # loaded bot object
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress tensorflow messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


""" Dataset Setup """

print('Preparing data.')
dataset_path = "Sentiment_Analysis_Dataset.csv"
DATASET_ENCODING = "ISO-8859-1"
input_length = 55

print("Opening file:", dataset_path)
df = pd.read_csv(dataset_path,
                 error_bad_lines=False,
                 encoding=DATASET_ENCODING,
                 dtype={"ItemID": int, "Sentiment": int, "SentimentSource": str, "SentimentText": str, },
                 warn_bad_lines=False, )

dat_len = len(df)
df = df[300000:800000]
# docs = df["SentimentText"]
# sentiments = df["Sentiment"]

# df2 = pd.read_csv('imdb_train.csv')

# print("Dataset size:", len(df))
# df2 = df2.sample(frac=1)

# docs2 = df["text"]
# sentiments2 = df["sentiment"]

main_df = df[['Sentiment', 'SentimentText']]
# temp_df = df2[['sentiment', 'text']]
# temp_df.columns = ['Sentiment', 'SentimentText']

# main_df = pd.concat((main_df, temp_df), axis=0, )
main_df = main_df.sample(frac=1)

# print(main_df)

docs = df["SentimentText"]
sentiments = df["Sentiment"]

# print(docs)
# print(sentiments)

# load GloVe data
print('...loading glove data.')
gd = glove_loader("./glove.twitter.27B/glove.twitter.27B.100d.txt")
print('...preparing tokenized data.')
inputs = gd.tokenize_data(docs, pad=input_length)
print('...preparing one hot labels.')
labels = gd.create_one_hot_labels(sentiments, num_classes=2)
print('...creating input tensors.')
inputs = gd.convert_to_tensor(inputs)
print('...creating label tensors.')
labels = gd.convert_to_simple_tensor(labels)


print("Creating embedding matrix.")
embedding_matrix = gd.words.values

num_words, num_word_dims = gd.words.shape[0], gd.words.shape[1],
print('Deleting unneeded variables.')
del df
del gd
del docs
del sentiments

print(inputs[0])
print(labels[0])
activation_fn = 'sigmoid'
dropout_factor = 0.2
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(num_words, num_word_dims,
#                               weights=[embedding_matrix],
#                               input_length=input_length,
#                               trainable=False),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_length,
#                                                        return_sequences=True,
#                                                        activation=activation_fn,
#                                                        kernel_initializer='normal')),
#     tf.keras.layers.Dropout(dropout_factor),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_length * 2,
#                                                        return_sequences=True,
#                                                        activation=activation_fn,
#                                                        kernel_initializer='normal')),
#     tf.keras.layers.Dropout(dropout_factor),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(input_length,
#                                                        activation=activation_fn,
#                                                        kernel_initializer='normal')),
#     tf.keras.layers.Dropout(dropout_factor),
#     tf.keras.layers.Dense(100, activation=activation_fn, kernel_initializer='normal'),
#     tf.keras.layers.Dropout(dropout_factor),
#     tf.keras.layers.Dense(60, activation=activation_fn, kernel_initializer='normal'),
#     tf.keras.layers.Dropout(dropout_factor),
#     tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='normal')
# ])

print(model.summary())

# compile the model
print("Compiling model.")
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.categorical_crossentropy,
#               metrics=['accuracy'])

# summarize the model
print("Training model.")

# Load old model for further training

# fit the model
# model.fit(inputs, labels, epochs=1, verbose=1, batch_size=1000)
model.fit(inputs, labels, epochs=1
          , verbose=1, batch_size=32)
# model.fit(inputs, labels, epochs=1, verbose=1, batch_size=50)
# save model
model.save("twitter_model_v4.h5")
# evaluate the model
print('Testing model.')
loss, accuracy = model.evaluate(inputs, labels, verbose=1)
print('Accuracy: %f' % (accuracy * 100))
