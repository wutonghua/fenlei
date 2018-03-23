#!/usr/bin/python
# -*- coding: utf-8 -*-
from yujv_process import process_line
from load_mysql import load_dataset
from GBDT_cidai import dafu_prediction
#数据匹配
disease_list=load_dataset("疾病库") #调用疾病库
question_list=[] #问答系列列表
line=question_list[-1] #调最新的一句问话
line=process_line(line) #语句预处理
for i in line:
	if i in disease_list:
		leibie,dafu=dafu_prediction(line)
		print(leibie + '\n' +dafu)
	else:
		question_list=question_list[:-1]
		for line1 in question_list[-1::-1]:
			line1=process_line(line1)
			for i in line1:
				if i in disease_list:
					line2=line.append(i)
					leibie, dafu = dafu_prediction(line2)
					print(leibie + '\n' + dafu)
					break
				else:
					print("不好意思没有找到您所说的疾病名称")
					disease_name=input("请输入您描述的疾病名称:")
					line3=line.append(disease_name)
					leibie,dafu=dafu_prediction(line3)
					print(leibie + '\n' + dafu)




