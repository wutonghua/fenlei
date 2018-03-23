#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

def load_dataset(name):
    datasets = {
        '治疗': 'zhiliao.csv',
        '病因': 'bingyin.csv',
        '症状': 'zhengzhuang.csv',
        '诊断': 'zhenduan.csv',
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('data', datasets[name])
    df = pd.read_csv(data_file, encoding='utf-8')
    # print('Number of reviews: {}'.format(len(df)))
    return df
def processing_null(file):

    NONE_VIN = (file["title"].isnull()) | (file["title"].apply(lambda x: str(x).isspace()))
    file_null = file[NONE_VIN]
    file_not_null = file[~NONE_VIN]
    return file_not_null['title']
def load_mysql(name):
    conn=MySQLdb.Connection(host='192.168.2.138', user='root', password='1234',
                              port=3306, database='medicine', charset='utf8')
    sql = "select * from %s " % name
    df=pd.read_sql(sql,conn)
    # print(df.head())
    return df
