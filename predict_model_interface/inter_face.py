#!/usr/bin/python
# -*- coding: utf-8 -*-
from ctb_mysql import CMySql
import jieba
import numpy as np
from sklearn.externals import joblib
#加载模型
text_classifier = joblib.load('text_classifier.pkl')
type_dict={'pathogeny':'sick_reason',
		   'symptom':'sick_manifestation',
		   'treatment':'sick_treat',
		   }

def class_identification(type,line,disease_list,answer):
	for i in line:
		if i in disease_list:
			select_sql="SELECT * FROM sick_tbl WHERE sick_name = '%s'" % (i)
			if type =='diagnosis':
				answer=answer+'diagnosis'
			else:
				reply = dbconn.queryone(select_sql)[type_dict.get(type)]
				answer = answer+reply
	return answer

def main(line,question_list):
	#实现多轮对话系统，增加问句的疾病种类并返回相应的答复
	disease_list=np.load('disease_list.npy')
	line1=jieba.lcut(line) #语句预处理

	question_list1=[]
	for i in question_list:
		question_list1.extend(jieba.cut(i))
	rel=[]
	for i in line1:
		if i in disease_list:
			rel.append(i)
	type = type_class
	answer = ''
	if len(rel) !=0:
		answer = class_identification(type, line1, disease_list, answer)
	else:
		if len(set(question_list1) & set(disease_list)) !=0:
			list = question_list[-1::-1]
			for line_src in list:
				line2 = jieba.lcut(line_src)
				# print("line2",line2)
				answer = class_identification(type, line2, disease_list, answer)
		else:
			print("不好意思,没有找到您所说的疾病名称,这种病暂时不在库中")
	return type,answer


if __name__=='__main__':
	dbconn = CMySql('192.168.2.138', '1234', 'root', 'medicine', 3306)
	while True:
		#输入语句
		line = input('请输入:')
		# 判断出所属类别
		content = text_classifier.process_line(line)
		type_class= text_classifier.predict(content)[0]
		question_list = ['头痛该如何治疗哦', '你好这种病怎么治疗哦']  # 问答系列列表与JAVA的接口
		type, answer=main(line,question_list)
		print(type + '\n'+ answer)





