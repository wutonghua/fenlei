#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import re
def load_positive_negative_data_files(bingyin_data_file, zhenduan_data_file,zhiliao_data_file,zhengzhuang_data_file):

    bingyin_examples = read_and_clean_zh_file(bingyin_data_file)
    zhenduan_examples = read_and_clean_zh_file(zhenduan_data_file)
    zhiliao_examples = read_and_clean_zh_file(zhiliao_data_file)
    zhengzhuang_examples = read_and_clean_zh_file(zhengzhuang_data_file)
    # Combine data
    x_text = bingyin_examples + zhenduan_examples + zhiliao_examples + zhengzhuang_examples
    # Generate labels
    bingyin_labels = [[1, 0,0,0] for _ in bingyin_examples]
    zhenduan_labels = [[0, 1,0,0] for _ in zhenduan_examples]
    zhiliao_labels = [[0, 0, 1, 0] for _ in zhiliao_examples]
    zhengzhuang_labels = [[0, 0, 0, 1] for _ in zhengzhuang_examples]

    y = np.concatenate([bingyin_labels , zhenduan_labels,zhiliao_labels,zhengzhuang_labels], 0)
    return [x_text, y]
def read_and_clean_zh_file(input_file, output_cleaned_file = None):
    lines = list(open(input_file, "r",encoding='utf-8').readlines())
    lines = [clean_str(seperate_line(line)) for line in lines]
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w') as f:
            for line in lines:
                f.write((line + '\n').encode('utf-8'))
    return lines

def seperate_line(line):
    return ''.join([word + ' ' for word in line])
def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
def padding_sentences(input_sentences, padding_token, padding_sentence_length = None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return (sentences, max_sentence_length)
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    Generate a batch iterator for a dataset
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx : end_idx]