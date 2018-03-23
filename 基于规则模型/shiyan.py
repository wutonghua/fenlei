#!/usr/bin/python
# -*- coding: utf-8 -*-
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
import pandas as pd

conn=MySQLdb.Connection(host='localhost', user='root', password='wujian',
						  port=3306, database='yixueziliao', charset='utf8')
cur=conn.cursor()
sql_find="select disease_name from disease_datebase"
cur.execute(sql_find)
list_disease=cur.fetchall()
for i in range(len(list_disease)):
	sql_find1="select disease_name from disease_datebase"
	cur.execute(sql_find1)
	list_disease1=cur.fetchall()
	list2=[]
	for j in range(len(list_disease1)):
		list2.append(list_disease1[j][0])
	print(list2)








