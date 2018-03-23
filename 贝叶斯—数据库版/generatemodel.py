#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import gensim
import jieba
import numpy as np
from load_mysql import load_dataset
from load_mysql import processing_null
#读取数据并预处理
df_bingyin_list=load_dataset('bingyin')
df_zhenduan_list=load_dataset('zhenduan')
df_zhengzhuang_list=load_dataset('zhengzhuang')
df_zhiliao_list=load_dataset('zhiliao')
stopwords=pd.read_csv('data/stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values  #导入停用词

#对各个类别数据进行空值符处理
df_bingyin_word=processing_null(df_bingyin_list)[0:1000]
# print(len(df_bingyin_word))
df_zhenduan_word=processing_null(df_zhenduan_list)[0:1000]
df_zhengzhuang_word=processing_null(df_zhengzhuang_list)[0:1000]
df_zhiliao_word=processing_null(df_zhiliao_list)[0:1000]
def build_sentence_vector(text, size, imdb_w2v):
	vec=np.zeros(size).reshape((1,size))
	count=0
	contents = jieba.lcut(text)
	for word in contents:
		if word not in stopwords:
			try:
				vec += imdb_w2v[word].reshape((1,size))
				count +=1
			except KeyError:
				continue
	if count !=0:
		vec /=count
	return vec
model_str = "song2vec.model"
model = gensim.models.Word2Vec.load(model_str)
imdb_w2v=model
n_dim=300
def get_train_vecs(x_train):
	#在训练集上建模
	train_vecs=np.concatenate([build_sentence_vector(z,n_dim,imdb_w2v) for z in x_train])
	return train_vecs
def get_line_vecs(line):
	train_vecs = [build_sentence_vector(line, n_dim, imdb_w2v)]
	return train_vecs
#把各个列表数据转化成word2vec形式
df_bingyin_word_vec=get_train_vecs(df_bingyin_word)
# print(df_bingyin_word_vec)
np.save('df_bingyin_word_vec',df_bingyin_word_vec)
df_zhenduan_word_vec=get_train_vecs(df_zhenduan_word)
np.save('df_zhenduan_word_vec',df_zhenduan_word_vec)
df_zhengzhuang_word_vec=get_train_vecs(df_zhengzhuang_word)
np.save('df_zhengzhuang_word_vec',df_zhengzhuang_word_vec)
df_zhiliao_word_vec=get_train_vecs(df_zhiliao_word)
np.save('df_zhiliao_word_vec',df_zhiliao_word_vec)



