#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
sess = tf.InteractiveSession()
saver = tf.train.Saver()
model=saver.restore(sess, tf.train.latest_checkpoint('models'))
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(model.accuracy, {model.input_x: nextBatch, model.input_y: nextBatchLabels})) * 100)