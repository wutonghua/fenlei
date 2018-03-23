#!/usr/bin/python
# -*- coding: utf-8 -*-
#导入常用的函数包
import random
import numpy
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from classifier import TextClassifier
from generatemodel import get_line_vecs
from load_dataset import load_dataset
from load_dataset import processing_null
from cos import ComputerNearestNeighbor
#读取数据并预处理
df_bingyin_list=load_dataset('病因')
df_zhenduan_list=load_dataset('诊断')
df_zhengzhuang_list=load_dataset('症状')
df_zhiliao_list=load_dataset('治疗')

#对各个类别数据进行空值符处理
df_bingyin_word=processing_null(df_bingyin_list)[0:1000]
df_zhenduan_word=processing_null(df_zhenduan_list)[0:1000]
df_zhengzhuang_word=processing_null(df_zhengzhuang_list)[0:1000]
df_zhiliao_word=processing_null(df_zhiliao_list)[0:1000]

bingyin = df_bingyin_word.values.tolist()
zhenduan=df_zhenduan_word.values.tolist()
zhengzhuang=df_zhengzhuang_word.values.tolist()
zhiliao=df_zhiliao_word.values.tolist()

#分别把各个类别数据整理成一个列表形式
sentences=[]
prep=preprocess(sentences,bingyin,zhenduan,zhengzhuang,zhiliao)
prep.preprocess_text(bingyin,sentences,'pathogeny')
prep.preprocess_text(zhenduan,sentences,'diagnosis')
prep.preprocess_text(zhengzhuang,sentences,'symptom')
prep.preprocess_text(zhiliao,sentences,'treatment')
random.shuffle(sentences)
#把各个列表数据转化成word2vec形式
df_bingyin_word_vec=numpy.load('df_bingyin_word_vec.npy').tolist()
df_zhenduan_word_vec=numpy.load('df_zhenduan_word_vec.npy').tolist()
df_zhengzhuang_word_vec=numpy.load('df_zhengzhuang_word_vec.npy').tolist()
df_zhiliao_word_vec=numpy.load('df_zhiliao_word_vec.npy').tolist()
#分割数据
x,y=zip(*sentences)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1234)

#训练数据，并训练各自的疾病数据成向量
text_classifier=TextClassifier()
text_classifier.fit(x_train,y_train)
print(text_classifier.score(x_test, y_test))
jibing_xl_dict={'diagnosis':df_zhenduan_word_vec,'treatment':df_zhiliao_word_vec,'symptom':df_zhengzhuang_word_vec,'pathogeny':df_bingyin_word_vec}

#输出预测类别
line=input('请输入:')
line=text_classifier.process_line(line)
leibie=text_classifier.predict(line)[0]
line_xl=get_line_vecs(line)[0].tolist()
# print(line_xl)
print(leibie)
pipei_xl=jibing_xl_dict.get(leibie,0)

#输出对应的诊断
nearest=ComputerNearestNeighbor(line_xl,pipei_xl)[0][1]
yuanshujv_dict = {'diagnosis':df_zhenduan_list,'treatment': df_zhiliao_list, 'symptom': df_zhengzhuang_list,
                  'pathogeny': df_bingyin_list}
pipei_shujv=yuanshujv_dict.get(leibie,0)
recommenddation=pipei_shujv.loc[nearest,'dafu1'] + pipei_shujv.loc[nearest,'dafu2']
print(recommenddation)







