#!/usr/bin/python
# -*- coding: utf-8 -*-
from xunlian_model import model_xunlian
from sklearn.externals import joblib
from precision_model import get_metrics,plot_confusion_matrix,fen_ge
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#加载模型
new_text_classifier=joblib.load('text_classifier.pkl')
#返回列表
bingyin_list,zhenduan_list,zhengzhuang_list,zhiliao_list,x_train,x_test,y_train,y_test=model_xunlian()
bingyin_list_x,bingyin_list_y=fen_ge(bingyin_list)
zhenduan_list_x,zhenduan_list_y=fen_ge(zhenduan_list)
zhengzhuang_list_x,zhengzhuang_list_y=fen_ge(zhengzhuang_list)
zhiliao_list_x,zhiliao_list_y=fen_ge(zhiliao_list)

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
#各个类别准确度与混淆矩阵
# y_predicted_counts=get_y_predicted_counts(zhiliao_list_x)
# accuracy, precision, recall, f1 = get_metrics(zhiliao_list_y, y_predicted_counts)
# print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
# cm = confusion_matrix(zhiliao_list_y, y_predicted_counts)
# fig = plt.figure(figsize=(10, 10))
# plot = plot_confusion_matrix(cm, classes=['pathogeny','diagnosis','symptom','treatment'], normalize=False, title='Confusion matrix')
# plt.show()
# print(cm)
if __name__=="__main__":
	precision_confusion(zhiliao_list_x,zhiliao_list_y)


