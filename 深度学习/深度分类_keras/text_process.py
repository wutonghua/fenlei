#!/usr/bin/python
# -*- coding: utf-8 -*-
from load_dataset import load_dataset
import pandas as pd
from sklearn.utils import shuffle
label_class = {
        '治疗': 0,
        '病因': 1,
        '症状': 2,
        '诊断': 3,
    }
stopwords=pd.read_csv('data/stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values  #导入停用词
#读取数据并预处理
df_bingyin=load_dataset('病因')
df_zhenduan=load_dataset('诊断')
df_zhengzhuang=load_dataset('症状')
df_zhiliao=load_dataset('治疗')
#只保留
def text_process(df,text_name):
    df['label']=label_class[text_name]
    df=df[['title','label','dafu1','dafu2']]
    return df
df_zhiliao=text_process(df_zhiliao,'治疗')
df_bingyin=text_process(df_bingyin,'病因')
df_zhenduan=text_process(df_zhenduan,'诊断')
df_zhengzhuang=text_process(df_zhengzhuang,'症状')
frames=[df_bingyin,df_zhenduan,df_zhengzhuang,df_zhiliao]
df=pd.concat(frames,axis=0, join='outer')
df=shuffle(df)
#去除空值
def delete_null(df):
    NONE_VIN = (df["title"].isnull()) | (df["title"].apply(lambda x: str(x).isspace()))
    df_null = df[NONE_VIN]
    df = df[~NONE_VIN]
    return df
df=delete_null(df)
# print(df.head())
def clean_str(df,text_field):
    df[text_field]=df[text_field].str.replace(r"[^\u4e00-\u9fff]", " ")
    df[text_field] = df[text_field].str.replace(r"\s{2,}", " ")
    return df
questions=clean_str(df,'title')
questions.to_csv('clean_data.csv')
clean_questions=pd.read_csv('clean_data.csv')





