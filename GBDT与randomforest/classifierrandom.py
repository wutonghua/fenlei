#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import pandas as pd
stopwords=pd.read_csv('data/stopwords.txt',index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values  #导入停用词
class TextClassifier():

    def __init__(self, classifier=RandomForestClassifier(n_estimators=100)):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,4), max_features=10000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)
    def process_line(self,x):
        segs = jieba.lcut(x)
        segs = filter(lambda x: len(x) > 1, segs)
        segs = filter(lambda x: x not in stopwords, segs)
        segs = " ".join(segs)
        return segs