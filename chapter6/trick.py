#!/usr/bin/env python
#-*-coding:utf-8-*-

import os,sys
sys.path.append(os.pardir)
from common import util
from dataset.mnist import load_mnist


#打乱数据集
(x_train,t_train),(x_test,t_test) = load_mnist()
x_train,t_train = shuffle_dataset(x_train,t_train)

#分割验证数据集
validation_rate = 0.20
validation_num = int(x_train.shape[0]*validation_rate)



