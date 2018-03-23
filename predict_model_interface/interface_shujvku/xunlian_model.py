#!/usr/bin/python
# -*- coding: utf-8 -*-
#导入常用的函数包
import random
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from preprocess import preprocess
from preprocess import preprocess2
from classifier import TextClassifier
from load_dataset import load_dataset
from load_dataset import processing_null
def model_xunlian():
	#读取数据并预处理
	df_bingyin_list=load_dataset('病因')
	df_zhenduan_list=load_dataset('诊断')
	df_zhengzhuang_list=load_dataset('症状')
	df_zhiliao_list=load_dataset('治疗')


	#对各个类别数据进行空值符处理
	df_bingyin_word=processing_null(df_bingyin_list)
	# print(len(df_bingyin_word))
	df_zhenduan_word=processing_null(df_zhenduan_list)
	df_zhengzhuang_word=processing_null(df_zhengzhuang_list)
	df_zhiliao_word=processing_null(df_zhiliao_list)

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

	# 分别把各个类别数据整理成各个列表形式
	bingyin_list = []
	zhenduan_list = []
	zhengzhuang_list = []
	zhiliao_list = []
	prep = preprocess2(bingyin_list,zhenduan_list,zhengzhuang_list,zhiliao_list, bingyin, zhenduan, zhengzhuang, zhiliao)
	prep.preprocess_lines(bingyin,bingyin_list,'pathogeny')
	prep.preprocess_lines(zhenduan,zhenduan_list,'diagnosis')
	prep.preprocess_lines(zhengzhuang,zhengzhuang_list,'symptom')
	prep.preprocess_lines(zhiliao,zhiliao_list,'treatment')


	#分割数据
	x,y=zip(*sentences)
	x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1234)

	#训练数据
	text_classifier=TextClassifier()
	text_classifier.fit(x_train,y_train)
	#保存并加载模型
	joblib.dump(text_classifier, 'text_classifier.pkl')
	# new_text_classifier=joblib.load('text_classifier.pkl')
	# precision=text_classifier.score(x_test, y_test)
	return bingyin_list,zhenduan_list,zhengzhuang_list,zhiliao_list,x_train,x_test,y_train,y_test
def process_shujv(df_zhenduan_list,df_bingyin_list,df_zhengzhuang_list,df_zhiliao_list):
	# 对各个类别数据进行空值符处理
	df_bingyin_word = processing_null(df_bingyin_list)
	# print(len(df_bingyin_word))
	df_zhenduan_word = processing_null(df_zhenduan_list)
	df_zhengzhuang_word = processing_null(df_zhengzhuang_list)
	df_zhiliao_word = processing_null(df_zhiliao_list)

	bingyin = df_bingyin_word.values.tolist()
	zhenduan = df_zhenduan_word.values.tolist()
	zhengzhuang = df_zhengzhuang_word.values.tolist()
	zhiliao = df_zhiliao_word.values.tolist()

	# 分别把各个类别数据整理成一个列表形式
	sentences = []
	prep = preprocess(sentences, bingyin, zhenduan, zhengzhuang, zhiliao)
	prep.preprocess_text(bingyin, sentences, 'pathogeny')
	prep.preprocess_text(zhenduan, sentences, 'diagnosis')
	prep.preprocess_text(zhengzhuang, sentences, 'symptom')
	prep.preprocess_text(zhiliao, sentences, 'treatment')
	random.shuffle(sentences)

	# 分别把各个类别数据整理成各个列表形式
	bingyin_list = []
	zhenduan_list = []
	zhengzhuang_list = []
	zhiliao_list = []
	prep = preprocess2(bingyin_list, zhenduan_list, zhengzhuang_list, zhiliao_list, bingyin, zhenduan, zhengzhuang,
					   zhiliao)
	prep.preprocess_lines(bingyin, bingyin_list, 'pathogeny')
	prep.preprocess_lines(zhenduan, zhenduan_list, 'diagnosis')
	prep.preprocess_lines(zhengzhuang, zhengzhuang_list, 'symptom')
	prep.preprocess_lines(zhiliao, zhiliao_list, 'treatment')

	# 分割数据
	x, y = zip(*sentences)
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

	# 训练数据
	text_classifier = TextClassifier()
	text_classifier.fit(x_train, y_train)
	# 保存并加载模型
	joblib.dump(text_classifier, 'text_classifier.pkl')
	# new_text_classifier=joblib.load('text_classifier.pkl')
	# precision=text_classifier.score(x_test, y_test)
	return x_train, x_test, y_train, y_test

