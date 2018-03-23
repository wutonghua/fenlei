#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from text_process import clean_str,delete_null
from load_dataset import fen_ci
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
#加载神经网络模型
model=load_model('keras_cnn.model')
#加载测试集
f=open('治疗.csv',encoding='utf-8')
df=pd.read_csv(f)
df=df[['title','label','dafu1','dafu2']]
#去除空值
df=delete_null(df)
clean_df=clean_str(df,'title')
#分词操作
clean_df=fen_ci(clean_df)
#数据预处理
EMBEDDING_DIM = 150
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS=14696
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(clean_df["text"].tolist())
sequences = tokenizer.texts_to_sequences(clean_df["text"].tolist())

word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

pred=model.predict(cnn_data)
id_class = {
        '0': '治疗',
        '1': '病因',
        '2': '症状',
        '3': '诊断',
    }
class_list=[]
number=0
for line in pred:
	line1=line.tolist()
	max_index=str(line1.index(max(line1)))
	if max_index=='3':
		number +=1
	class_list.append(id_class.get(max_index))
print("诊断类型的准确率为%f" % (number/len(class_list)))



