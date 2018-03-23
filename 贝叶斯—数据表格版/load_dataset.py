#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
def load_dataset(name):
    datasets = {
        '治疗': '治疗.csv',
        '病因': '病因.csv',
        '症状': '症状.csv',
        '诊断': '诊断.csv',
    }
    if name not in datasets:
        raise ValueError(name)
    data_file = os.path.join('data', datasets[name])
    df = pd.read_csv(data_file, encoding='utf-8')
    # print('Number of reviews: {}'.format(len(df)))
    return df
def processing_null(file):

    NONE_VIN = (file["title"].isnull()) | (file["title"].apply(lambda x: str(x).isspace()))
    file_null = file[NONE_VIN]
    file_not_null = file[~NONE_VIN]
    return file_not_null['title']
