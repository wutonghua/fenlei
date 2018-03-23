#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import pandas as pd
from load_dataset import processing_null
import os
import csv
#数据预处理
def test(line):
	text_classifier = joblib.load('text_classifier.pkl')
	content=text_classifier.process_line(line)
	leibie=text_classifier.predict(content)[0]
	return leibie

#文本处理，训练并随时存储数据
path="C:\\Users\\Administrator\\Desktop\\GBDT_predicted\\ziliao"
folder_list=os.listdir(path)
with open('new_txt.csv', 'w',newline='') as f:
	writer = csv.writer(f)
	writer.writerow(['line', 'leibie'])
	for folder in folder_list:
		folder_path=os.path.join(path,folder)
		df = pd.read_csv(folder_path, encoding='gbk')
		# df = pd.read_csv('ziliao/aixiao.csv', encoding='gbk')
		df = processing_null(df)
		for line in df:
			# print(type(line))
			liebie = test(line)
			# new = pd.DataFrame([{'title': line,'leibie':liebie}], index=['0'])
			writer.writerow([line,liebie])
f.close()

