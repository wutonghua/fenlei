#!/usr/bin/python
# -*- coding: utf-8 -*-
from load_dataset import shujv_zhengli
#加载数据
df_bingyin=shujv_zhengli('bingyin.txt','病因')
df_zhiliao=shujv_zhengli('zhiliao.txt','治疗')
df_zhengzhuang=shujv_zhengli('zhengzhuang.txt','症状')
df_zhenduan=shujv_zhengli('zhenduan.txt','诊断')
flist = ['bingyin.txt','zhenduan.txt','zhengzhuang.txt','zhiliao.txt']
# #要写入的文件
# ofile = open('yixue.txt', 'w')
# #遍历读取所有文件，并写入到输出文件
# for fr in flist:
#     for txt in open(fr, 'r',encoding='utf-8'):
#         ofile.write(txt)
# ofile.close()



