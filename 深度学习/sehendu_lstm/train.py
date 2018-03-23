#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import word2vec_helpers
import data_helper
from text_lstm import TextLSTM


# Load data
print("Loading data...")
x_text, y = data_helper.load_positive_negative_data_files('bingyin.txt', 'zhenduan.txt', 'zhiliao.txt', 'zhengzhuang.txt')

# Get embedding vector
embedding_dim=300
sentences, max_document_length = data_helper.padding_sentences(x_text, '<PADDING>')
x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = embedding_dim, file_to_save ='trained_word2vec.model'))
print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))


# Shuffle data randomly
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(0.1 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

batchSize = 24
lstmUnits = 64
numClasses = 4
iterations = 50000
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
with sess.as_default():
	saver = tf.train.Saver()
	lstm=TextLSTM(max_document_length,batchSize,numClasses,embedding_dim,x)
	optimizer = tf.train.AdamOptimizer().minimize(lstm.loss)
	for i in range(iterations):
		# Next Batch of reviews
		batches = data_helper.batch_iter(list(zip(x_train, y_train)), batchSize, iterations)
		for batch in batches:
			nextBatch, nextBatchLabels = zip(*batch)
			sess.run(optimizer, {lstm.input_x: nextBatch, lstm.input_y: nextBatchLabels})

			if (i % 1000 == 0 and i != 0):
				loss_ = sess.run(lstm.loss, {lstm.input_x: nextBatch, lstm.input_y: nextBatchLabels})
				accuracy_ = sess.run(lstm.accuracy, {lstm.input_x: nextBatch, lstm.input_y: nextBatchLabels})

				print("iteration {}/{}...".format(i + 1, iterations),
					  "loss {}...".format(loss_),
					  "accuracy {}...".format(accuracy_))
			# Save the network every 10,000 training iterations
			if (i % 10000 == 0 and i != 0):
				save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
				print("saved to %s" % save_path)


