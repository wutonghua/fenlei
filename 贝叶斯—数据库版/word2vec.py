#!/usr/bin/python
# -*- coding: utf-8 -*-
from load_mysql import load_dataset
from load_mysql import processing_null
import pandas as pd
import gensim
import jieba
from random import shuffle
import multiprocessing
stopwords=pd.read_csv('data/stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values  #导入停用词
#读取数据并预处理
df_bingyin=load_dataset('bingyin')
df_zhenduan=load_dataset('zhenduan')
df_zhengzhuang=load_dataset('zhengzhuang')
df_zhiliao=load_dataset('zhiliao')

df_bingyin=processing_null(df_bingyin)[0:1000]
df_zhenduan=processing_null(df_zhenduan)[0:1000]
df_zhengzhuang=processing_null(df_zhengzhuang)[0:1000]
df_zhiliao=processing_null(df_zhiliao)[0:1000]
frames=[df_bingyin,df_zhenduan,df_zhengzhuang,df_zhiliao]
df=pd.concat(frames,axis=0, join='outer')
# print(len(df))
# print(df.head())
# for line in df:
# 	print(line)

def parse_titlelist_get_sequence(in_line, titlelist_sequence):
	title_sequence = []
	# print("****",in_line)
	contents = jieba.lcut(in_line)
	# 解析title序列
	for title in contents:
		if title not in stopwords:
			try:
				title_sequence.append(title)
			except:
				print("title format error")
				print(title+"\n")
	for i in range(len(title_sequence)):
		shuffle(title_sequence)
		titlelist_sequence.append(title_sequence)
def train_ttle2vec(in_file, out_file):
	#所有问题序列
	titlelist_sequence = []
	#遍历所有title
	for line in in_file:
		# if float(isnan(line)) == True:
		# 	continue
		parse_titlelist_get_sequence(line,titlelist_sequence)
#使用word2vec训练
	cores = multiprocessing.cpu_count()
	print("using all "+str(cores)+" cores")
	print("Training word2vec model...")
	model = gensim.models.Word2Vec(sentences=titlelist_sequence, size=300, min_count=2, window=5, workers=cores)
	print("Saving model...")
	model.save(out_file)
# for line in df['title']:
# 	if line ==',title':
# 		continue
# 	print(line)
model_file = "song2vec.model"
train_ttle2vec(df, model_file)
model_str = "song2vec.model"
model = gensim.models.Word2Vec.load(model_str)
# print(model['疾病'])


