#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
import pandas as pd

def load_dataset(name):
	conn=MySQLdb.Connection(host='localhost', user='root', password='wujian',
							  port=3306, database='yixueziliao', charset='utf8')
	sql = "select * from %s " % name
	df=pd.read_sql(sql,conn)
	# print(df.head())
	return df
# if __name__=='__main__':
# 	load_dataset('bingyin')
def processing_null(file):
	NONE_VIN=(file['title'].isnull()) | (file['title'].apply(lambda x:str(x).isspace()))
	file_null=file[NONE_VIN]
	file_not_null=file[~NONE_VIN]
	return file_not_null['title']