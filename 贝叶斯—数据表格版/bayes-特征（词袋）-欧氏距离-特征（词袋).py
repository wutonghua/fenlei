#!/usr/bin/python
# -*- coding: utf-8 -*-
#导入常用的函数包
import random
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from preprocess import preprocess
from preprocess import preprocess1
from classifier import TextClassifier
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
# print(len(df_bingyin_word))
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

#分别把各个类别数据整理成各个列表形式
bingyin_list=[]
zhenduan_list=[]
zhengzhuang_list=[]
zhiliao_list=[]
prep=preprocess1(bingyin_list,zhenduan_list,zhengzhuang_list,zhiliao_list,bingyin,zhenduan,zhengzhuang,zhiliao)
prep.preprocess_lines(bingyin,bingyin_list)
prep.preprocess_lines(zhenduan,zhenduan_list)
prep.preprocess_lines(zhengzhuang,zhengzhuang_list)
prep.preprocess_lines(zhiliao,zhiliao_list)

#分割数据
x,y=zip(*sentences)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1234)

#训练数据
text_classifier=TextClassifier()
text_classifier.fit(x_train,y_train)
#保存并加载模型
joblib.dump(text_classifier, 'text_classifier.pkl')
new_text_classifier=joblib.load('text_classifier.pkl')
print(new_text_classifier.score(x_test, y_test))
bingyin_xl=new_text_classifier.features(bingyin_list).todense()
zhiliao_xl=new_text_classifier.features(zhiliao_list).todense()
zhengzhuang_xl=new_text_classifier.features(zhengzhuang_list).todense()
zhenduan_xl=new_text_classifier.features(zhenduan_list).todense()
jibing_xl_dict={'diagnosis':zhenduan_xl,'treatment':zhiliao_xl,'symptom':zhengzhuang_xl,'pathogeny':bingyin_xl}
# print(zhenduan_xl)

#输出预测类别
line=input('请输入:')
line=new_text_classifier.process_line(line)
leibie=new_text_classifier.predict(line)[0]
line_xl=new_text_classifier.features([line]).todense()
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







