#!/usr/bin/python
# -*- coding: utf-8 -*-
import gensim
import numpy as np
import pandas as pd
#加载word2vec模型
model_str = "yixue2vec.model"
word2vec = gensim.models.Word2Vec.load(model_str)
clean_questions=pd.read_csv('clean_data.csv')
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=150):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['token'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                                generate_missing=generate_missing))
    return list(embeddings)
list_labels = clean_questions["label"].tolist()
embeddings = get_word2vec_embeddings(word2vec, clean_questions)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels,test_size=0.2, random_state=40)
