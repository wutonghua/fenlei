#!/usr/bin/python
# -*- coding: utf-8 -*-
import jieba
import pandas as pd
stopwords=pd.read_csv('data/stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values  #导入停用词
def process_line(x):
	segs = jieba.lcut(x)
	segs = filter(lambda x: len(x) > 1, segs)
	segs = filter(lambda x: x not in stopwords, segs)
	segs = list(" ".join(segs))
	for i in range(segs.count(' ')):
		segs.remove(' ')
	return segs
if __name__ == "__main__":
	m= process_line("我来北京了")
	# for i in range(m.count(' ')):
	# 	m.remove(' ')
	print(m)