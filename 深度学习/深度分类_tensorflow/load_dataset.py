#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import jieba
def load_dataset(name):
    datasets = {
        '治疗': '治疗.csv',
        '病因': '病因.csv',
        '症状': '症状.csv',
        '诊断': '诊断.csv',
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('data', datasets[name])
    f=open(data_file, encoding='utf-8')
    df = pd.read_csv(f)
    # print('Number of reviews: {}'.format(len(df)))
    return df
label_class = {
        '治疗': 0,
        '病因': 1,
        '症状': 2,
        '诊断': 3,
    }
def processing_null(file):

    NONE_VIN = (file["title"].isnull()) | (file["title"].apply(lambda x: str(x).isspace()))
    file_null = file[NONE_VIN]
    file_not_null = file[~NONE_VIN]
    return file_not_null
def fen_ci(df):
    cw = lambda x: ' '.join(jieba.cut(x))
    cw1=lambda x: jieba.lcut(x)
    df['token'] = df['title'].apply(cw1)
    df['text'] = df['title'].apply(cw)
    return df
def text_process(df,text_name):
    df['label']=label_class[text_name]
    df=df[['title','label','dafu1','dafu2']]
    return df
def shujv_zhengli(file,label):
    df=load_dataset(label)
    df=text_process(df,label)
    df=processing_null(df)
    df['title'].to_csv(file, encoding='utf-8',index=False, header=False)
    return df
