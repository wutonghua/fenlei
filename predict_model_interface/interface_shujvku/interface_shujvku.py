#!/usr/bin/python
# -*- coding: utf-8 -*-
from predict_model import predicted_leibie
from predicted_plot import precision_confusion
from load_dataset import load_mysql
from xunlian_model import process_shujv
import jieba
import numpy as np
#输入问句
line=input('请输入:')
#判断出所属类别并存入相应数据表
leibie,df_zhenduan_list,df_bingyin_list,df_zhengzhuang_list,df_zhiliao_list=predicted_leibie(line)
#用新数据进行模型的训练
# x_train, x_test, y_train, y_test=process_shujv(df_zhenduan_list,df_bingyin_list,df_zhengzhuang_list,df_zhiliao_list)
x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')

#实现多轮对话系统，增加问句的疾病种类并返回相应的答复
disease_base=load_mysql("sick_tbl")
# print(disease_base.head())
# disease_list= list(disease_base['sick_name'])  #调用疾病库
# np.save('disease_list.npy',disease_list)
disease_list=np.load('disease_list.npy')
# print(disease_list[:5])
question_list=['头痛该如何治疗哦','你好这种病怎么治疗哦'] #问答系列列表与JAVA的接口
line1=jieba.lcut(line) #语句预处理
# print(line1)
question_list1=[]
for i in question_list:
	question_list1.extend(jieba.cut(i))
rel=[]
for i in line1:
	if i in disease_list:
		rel.append(i)
def main():
	if len(rel) !=0:
		for i in line1:
			if i in disease_list:
				dafu=((disease_base.loc[(disease_base['sick_name'] == str(i)),['sick_treat']]).values)[0][0]
				# print(dafu)
				print(leibie + '\n' + dafu)
				print(precision_confusion(x_test, y_test))
				break
	else:
		if len(set(question_list1) & set(disease_list)) !=0:
			list = question_list[-1::-1]
			# print(list)
			for line_src in list:
				line2 = jieba.lcut(line_src)
				# print("line2",line2)
				for j in line2:
					if j in disease_list:
						dafu = ((disease_base.loc[(disease_base['sick_name'] == str(j)),['sick_treat']]).values)[0][0]
						print(leibie + '\n' + dafu)
						print(precision_confusion(x_test, y_test))
						break
		else:
			print("不好意思没有找到您所说的疾病名称")
			disease_name=input("请输入您描述的疾病名称:")
			dafu = ((disease_base.loc[(disease_base['sick_name'] == disease_name),['sick_treat']]).values)[0][0]
			print(leibie + '\n' + dafu)
			print(precision_confusion(x_test, y_test))


if __name__=='__main__':
	main()