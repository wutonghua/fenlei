#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import pandas as pd
from load_dataset import processing_null
from load_dataset import load_dataset
import os
from ctb_mysql import CMySql
#数据预处理
def test(line):
	text_classifier = joblib.load('text_classifier.pkl')
	content=text_classifier.process_line(line)
	leibie=text_classifier.predict(content)[0]
	return leibie

# #加载数据
# df_bingyin_list=load_dataset('病因')
# df_zhenduan_list=load_dataset('诊断')
# df_zhengzhuang_list=load_dataset('症状')
# df_zhiliao_list=load_dataset('治疗')
# #文本处理，训练并随时存储到数据表格中
# path="C:\\Users\\Administrator\\Desktop\\predicted_model\\ziliao"
# folder_list=os.listdir(path)
# for folder in folder_list:
# 	folder_path=os.path.join(path,folder)
# 	df = pd.read_csv(folder_path, encoding='gbk')
# 	# df = pd.read_csv('ziliao/aixiao.csv', encoding='gbk')
# 	df = processing_null(df)
# 	for line in df:
# 		# print(type(line))
# 		liebie = test(line)
# 		new = pd.DataFrame([{'title': line}], index=['0'])
# 		# print(liebie)
# 		if liebie =='diagnosis':
# 			df_zhenduan_list=df_zhenduan_list.append(new,ignore_index=True)
# 			df_zhenduan_list.to_csv("data/诊断.csv",encoding='utf-8')
# 		elif liebie=='pathogeny':
# 			df_bingyin_list = df_bingyin_list.append(new, ignore_index=True)
# 			df_bingyin_list.to_csv("data/病因.csv", encoding='utf-8')
# 		elif liebie=='symptom':
# 			df_zhengzhuang_list = df_zhengzhuang_list.append(new, ignore_index=True)
# 			df_zhengzhuang_list.to_csv("data/症状.csv",encoding='utf-8')
# 		else:
# 			df_zhiliao_list = df_zhiliao_list.append(new, ignore_index=True)
# 			df_zhiliao_list.to_csv("data/治疗.csv", encoding='utf-8')
#文本处理，训练并随时存储到数据表格中
def predicted_leibie(line):
	# 加载数据
	df_bingyin_list = load_dataset('病因')
	df_zhenduan_list = load_dataset('诊断')
	df_zhengzhuang_list = load_dataset('症状')
	df_zhiliao_list = load_dataset('治疗')
	liebie = test(line)
	new = pd.DataFrame([{'title': line}], index=['0'])
	# print(liebie)
	if liebie == 'diagnosis':
		df_zhenduan_list = df_zhenduan_list.append(new, ignore_index=True)
		df_zhenduan_list.to_csv("data/zhenduan.csv", encoding='utf-8')
	elif liebie == 'pathogeny':
		df_bingyin_list = df_bingyin_list.append(new, ignore_index=True)
		df_bingyin_list.to_csv("data/bingyin.csv", encoding='utf-8')
	elif liebie == 'symptom':
		df_zhengzhuang_list = df_zhengzhuang_list.append(new, ignore_index=True)
		df_zhengzhuang_list.to_csv("data/zhengzhuang.csv", encoding='utf-8')
	else:
		df_zhiliao_list = df_zhiliao_list.append(new, ignore_index=True)
		df_zhiliao_list.to_csv("data/zhiliao.csv", encoding='utf-8')
	return liebie,df_zhenduan_list,df_bingyin_list,df_zhengzhuang_list,df_zhiliao_list
#文本处理，训练并随时存储到数据库中
def predicted_leibie_shujvku(line):
	# 加载数据库
	dbconn = CMySql('localhost', 'wujian', 'root', 'yixueziliao', 3306)
	liebie = test(line)
	diagnosis_num=0
	pathogeny_num=0
	symptom_num=0
	treat_num=0
	if liebie == 'diagnosis':
		sql="insert into zhenduan ( title) values ('%s')" % (line)
		diagnosis_num = diagnosis_num+dbconn.insert(sql)
	elif liebie == 'pathogeny':
		sql = "insert into bingyin ( title) values ('%s')" % (line)
		pathogeny_num = pathogeny_num+dbconn.insert(sql)
	elif liebie == 'symptom':
		sql = "insert into zhengzhuang ( title) values ('%s')" % (line)
		symptom_num = symptom_num+dbconn.insert(sql)
	else:
		sql = "insert into zhiliao ( title) values ('%s')" % (line)
		treat_num = treat_num+dbconn.insert(sql)
	return liebie,diagnosis_num,pathogeny_num,symptom_num,treat_num












