#!/usr/bin/python
# -*- coding: utf-8 -*-
import jieba
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import gensim
from load_dataset import fen_ci
clean_questions=pd.read_csv('clean_data.csv')
#分词操作
clean_questions=fen_ci(clean_questions)
# print(clean_questions.head())
all_words=[word for tokens in clean_questions['token'] for word in tokens]
sentence_lengths=[len(tokens) for tokens in clean_questions['token']]
VOCAB=list(set(all_words))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))
#画图看各个长度的比例占比
import matplotlib.pyplot as plt
# fig=plt.figure(figsize=(10,10))
# plt.xlabel('sentence length')
# plt.ylabel('number of sentences')
# plt.hist(sentence_lengths)
# plt.show()
#加载word2vec模型
model_str = "yixue2vec.model"
word2vec = gensim.models.Word2Vec.load(model_str)
#数据集准备
EMBEDDING_DIM = 150
MAX_SEQUENCE_LENGTH = 50
VOCAB_SIZE = len(VOCAB)
VALIDATION_SPLIT=.2
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(clean_questions["text"].tolist())
sequences = tokenizer.texts_to_sequences(clean_questions["text"].tolist())

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(clean_questions["label"]))

indices=np.arange(cnn_data.shape[0])
np.random.shuffle(indices)
cnn_data=cnn_data[indices]
labels=labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

embedding_weights=np.zeros((len(word_index) +1,EMBEDDING_DIM))
for word,index in word_index.items():
    embedding_weights[index,:]=word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(embedding_weights.shape)
#模型搭建
from keras.layers import Dense,Input,Flatten,Dropout,Merge
from keras.layers import Conv1D,MaxPooling1D,Embedding
from keras.layers import LSTM,Bidirectional
from keras.models import Model


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)

    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv == True:
        x = Dropout(0.5)(l_merge)
    else:
        # Original Yoon Kim model
        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)

    preds = Dense(labels_index, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return model
#训练模型
x_train = cnn_data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = cnn_data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

model = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index)+1, EMBEDDING_DIM,
                len(list(clean_questions["label"].unique())), False)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=128)
model.save('keras_cnn.model')
