#!/usr/bin/python
# -*- coding: utf-8 -*-
try:
	import pymysql
except ImportError:
	raise ImportError("[E]:pymysql module not found")

class CMySql(object):
	def __init__(self,host,pwd,user,db,port=3306):
		self.Option={"host":host,"password":pwd,"username":user,"database":db,"port":port}
		self.__connection__()
	def __del__(self):
		if self.__conn:self.close()
	def __connection__(self):
		try:
			self.__conn=pymysql.connect(host=self.Option['host'],user=self.Option['username'],
										passwd=self.Option['password'],db=self.Option['database'],port=self.Option['port'],charset="utf8")
			self.__dictcursor=pymysql.cursors.DictCursor
		except Exception as e:
			print(e)
			raise Exception("[E] Cannot connect to %s" % self.Option["host"])
	def execute(self,sqlstate):
		self.cursor=self.__conn.cursor()
		self.cursor.execute(sqlstate)
		self.commit()
	def insert(self,sqlstate):
		self.cursor=self.__conn.cursor()
		self.cursor.execute(sqlstate)
		lastinsertid=int(self.__conn.insert_id())
		self.commit()
		return lastinsertid
	def query(self,sqlstate):
		self.cursor=self.__conn.cursor(self.__dictcursor)
		self.cursor.execute(sqlstate)
		return self.cursor.fetchall()
	def querylist(self,sqlstate,size):
		self.cursor=self.__conn.cursor()
		self.cursor.execute(sqlstate)
		return self.cursor.fetchmany(size)
	def queryone(self,sqlstate):
		self.cursor=self.__conn.cursor(self.__dictcursor)
		self.cursor.execute(sqlstate)
		return self.cursor.fetchone()
	def close(self):
		self.__conn.close()
	def commit(self):
		try:
			self.__conn.commit()
		except:
			self.__conn.rollback()
			raise

