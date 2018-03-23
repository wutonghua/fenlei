#!/usr/bin/python
# -*- coding: utf-8 -*-
import jieba
import pandas as pd
stopwords=pd.read_csv('data/stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values  #导入停用词
class preprocess(object):
    def __init__(self,sentences,bingyin,zhenduan,zhengzhuang,zhiliao):
        self.sentences=[]
        self.bingyin_list=[]
        self.zhenduan_list = []
        self.zhengzhuang_list = []
        self.zhiliao_list = []
        self.bingyin=bingyin
        self.zhenduan=zhenduan
        self.zhengzhuang=zhengzhuang
        self.zhiliao=zhiliao
    def preprocess_text(self,title_lines,sentences,category):
        for line in title_lines:
            try:
                segs=jieba.lcut(line)
                segs=filter(lambda x:len(x) >1,segs)
                segs=filter(lambda x:x not in stopwords,segs)
                sentences.append((" ".join(segs),category))
            except Exception as e:
                print(line)
                continue
class preprocess1(object):
    def __init__(self, bingyin_list,zhenduan_list,zhengzhuang_list,zhiliao_list, bingyin, zhenduan, zhengzhuang, zhiliao):
        self.bingyin_list = []
        self.zhenduan_list = []
        self.zhengzhuang_list = []
        self.zhiliao_list = []
        self.bingyin = bingyin
        self.zhenduan = zhenduan
        self.zhengzhuang = zhengzhuang
        self.zhiliao = zhiliao
    def preprocess_lines(self,title_lines,sentences):
        for line in title_lines:
            try:
                segs=jieba.lcut(line)
                segs=filter(lambda x:len(x) >1,segs)
                segs=filter(lambda x:x not in stopwords,segs)
                sentences.append((" ".join(segs)))
            except Exception as e:
                print(line)
                continue
