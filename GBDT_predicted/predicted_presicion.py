#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.externals import joblib
from precision_model import get_metrics,plot_confusion_matrix,fen_ge
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
#加载模型
new_text_classifier=joblib.load('text_classifier.pkl')
#得到预测的数值
def get_y_predicted_counts(x_test):
	y_predicted_counts = []
	for line in x_test:
		leibie = new_text_classifier.predict(line)[0]
		y_predicted_counts.append(leibie)
	return y_predicted_counts
#整体准确度与混淆矩阵
def precision_confusion(x_test,y_test):
	y_predicted_counts=get_y_predicted_counts(x_test)
	accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
	print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
	cm = confusion_matrix(y_test, y_predicted_counts)
	fig = plt.figure(figsize=(10, 10))
	plot = plot_confusion_matrix(cm, classes=['diagnosis','pathogeny','symptom','treatment'], normalize=False, title='Confusion matrix')
	plt.show()
	print(cm)
df=pd.read_csv('new_txt.csv',encoding='utf-8')
df_train=list(df['line'].values.astype('U'))
df_test=list(df['leibie'])
print(df.groupby('leibie').count())
if __name__=='__main__':
	precision_confusion(df_train, df_test)
