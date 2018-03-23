#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
class TextLSTM(object):
	def __init__(self,maxSeqLength,batchSize,numClasses,embedding_size,wordVectors):
		lstmUnits = 64
		self.input_y=tf.placeholder(tf.float32, [batchSize, numClasses])
		self.input_x = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
		self.data = tf.Variable(tf.zeros([batchSize, maxSeqLength, embedding_size]), dtype=tf.float32)
		self.data = tf.nn.embedding_lookup(wordVectors, self.input_x)
		lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
		lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
		value, _ = tf.nn.dynamic_rnn(lstmCell, self.data, dtype=tf.float32)

		weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
		bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
		value = tf.transpose(value, [1, 0, 2])
		# 取最终的结果值
		last = tf.gather(value, int(value.get_shape()[0]) - 1)
		self.prediction = (tf.matmul(last, weight) + bias)
		correctPred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.input_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.input_y))

