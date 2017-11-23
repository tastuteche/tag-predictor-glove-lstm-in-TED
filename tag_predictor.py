import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss, f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from joblib import Parallel, delayed

b_dir = './glove-global-vectors-for-word-representation/'
b2_dir = './ted-talks/'

# preparing embedding index
embeddings_index = {}
with open(b_dir + 'glove.6B.200d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


speaker_gender = pd.read_csv('ted_speaker_gender.csv')
df = pd.read_csv(b2_dir + 'ted_main.csv')
df_transcript = pd.read_csv(b2_dir + 'transcripts.csv')

df_with_gender = pd.merge(df, speaker_gender, how='inner',
                          left_on='main_speaker', right_on='name_in_profile')

df_with_gender = pd.merge(df_with_gender, df_transcript, how='inner',
                          left_on='url', right_on='url')

df_with_gender['tags_eval'] = df_with_gender['tags'].apply(
    lambda x: ast.literal_eval(x))

df_with_gender['he_she_count'] = df_with_gender['he_count'] > df_with_gender['she_count']


def preprocess_functions(x):
    return x.lower()


df_with_gender['title_changed'] = df_with_gender['title'].apply(
    lambda x: preprocess_functions(x))

# prepare comments section
df_with_gender['description_changed'] = df_with_gender['transcript'].apply(
    lambda x: preprocess_functions(x))

# text for training Keras Tokenizer (includes comments and titles)
df_with_gender['for_tokenizer'] = df_with_gender['title_changed'] + \
    " " + df_with_gender['description_changed']
df_with_gender['for_tokenizer'][0][:500]


df_with_gender['title_changed'].str.split().apply(len).describe()
df_with_gender['description_changed'].str.split().apply(len).describe()

# initialize parameters for embedding layer
max_num_words = 20000
max_sequence_length_text = 2700
max_sequence_length_title = 10
val_split = 0.10
embedding_dim = 200


# train tokenizer
tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(df_with_gender['for_tokenizer'].values.tolist())
word_index = tokenizer.word_index

# tokenization of comments
sq_comments = tokenizer.texts_to_sequences(
    df_with_gender['description_changed'].values.tolist())
x_comm = pad_sequences(sq_comments, maxlen=max_sequence_length_text)
x_comm.shape

# tokenization of titles
sq_titles = tokenizer.texts_to_sequences(
    df_with_gender['title_changed'].values.tolist())
x_titles = pad_sequences(sq_titles, maxlen=max_sequence_length_title)
x_titles.shape

# labels (or tags)
labeller = preprocessing.MultiLabelBinarizer()
labels = labeller.fit_transform(
    df_with_gender['tags_eval'])
labels.shape

# separate the data for train and test split
x_comm_tr, x_comm_val, x_tl_tr, x_tl_val, y_tr, y_val = train_test_split(x_comm, x_titles, labels,
                                                                         test_size=0.1, random_state=12)
y_val.shape

# preparing embedding matrix and layer
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# embedding layer for comments
embedding_layer_comm = Embedding(len(word_index) + 1,
                                 embedding_dim,
                                 weights=[embedding_matrix],
                                 input_length=max_sequence_length_text,
                                 trainable=False)

# embedding layer for titles
embedding_layer_title = Embedding(len(word_index) + 1,
                                  embedding_dim,
                                  weights=[embedding_matrix],
                                  input_length=max_sequence_length_title,
                                  trainable=False)


# Keras neural network
# thanks to https://keras.io/getting-started/functional-api-guide/
# thanks to https://www.kaggle.com/lystdo/lb-0-18-lstm-with-glove-and-magic-features

comm_input = Input(shape=(max_sequence_length_text,), dtype='int32')
emb_seq_1 = embedding_layer_comm(comm_input)
conv_1 = Conv1D(filters=334,
                kernel_size=30,
                strides=5,
                padding='valid',
                activation='relu',)(emb_seq_1)
conv_1 = MaxPooling1D(pool_size=10)(conv_1)
comm = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(conv_1)

title_input = Input(shape=(max_sequence_length_title,), dtype='int32')
emb_seq_2 = embedding_layer_title(title_input)
title = LSTM(6, dropout=0.2, recurrent_dropout=0.2)(emb_seq_2)

merged = concatenate([comm, title])
merged = BatchNormalization()(merged)
merged = Dropout(0.2)(merged)

merged = Dense(300, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.2)(merged)

preds = Dense(labels.shape[1], activation='sigmoid')(merged)

model = Model(inputs=[comm_input, title_input], outputs=preds)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

history = model.fit([x_comm_tr, x_tl_tr], y_tr,
                    validation_data=([x_comm_val, x_tl_val], y_val),
                    epochs=10, batch_size=10)

import matplotlib.pyplot as plt
# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
plt.savefig('tag_train_loss.png', dpi=200)
plt.clf()
plt.cla()
plt.close()

y_pred = model.predict([x_comm_val, x_tl_val])
y_pred_binary = y_pred
y_pred_binary[y_pred_binary > 0.05] = 1
y_pred_binary[y_pred_binary <= 0.05] = 0
y_pred_inverse = labeller.inverse_transform(y_pred_binary)
y_val_inverse = labeller.inverse_transform(y_val)

y_pred_inverse[3]
hamming_loss(y_pred, y_val)


# selecting best thresholds for labels
# https://github.com/viig99/stackexchange-transfer-learning/blob/master/kaggle_top1.py

threshold = np.arange(0.0001, 0.02, 0.0005)
y_pred = model.predict([x_comm_val, x_tl_val])
output = np.array(y_pred)


def bestThreshold(y_prob, threshold, i):
    acc = []
    for j in threshold:
        y_predicted = np.greater_equal(y_prob, j) * 1
        acc.append(matthews_corrcoef(y_val[:, i], y_predicted))
    acc = np.array(acc)
    index = np.where(acc == np.max(acc.max()))
    return threshold[index[0][0]]


best_threshold = Parallel(n_jobs=4, verbose=1)(delayed(bestThreshold)(
    output[:, i], threshold, i) for i in range(output.shape[1]))

y_pred_udb = np.greater_equal(output, np.array(best_threshold)).astype(np.int8)
y_pred_udb_inverse = labeller.inverse_transform(y_pred_udb)

output3 = np.array(y_pred)
idx3 = output3.argsort(axis=1)
for i in range(0, idx3.shape[0]):
    output3[i, idx3[i, :-8]] = 0
    output3[i, idx3[i, -8:]] = 1
top3_inverse = labeller.inverse_transform(output3)
