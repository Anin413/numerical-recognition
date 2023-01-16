import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#导入库文件 import
(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
#导入MNIST数据集
train_images.shape
train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)
#维度变换
train_images.shape
train_images = train_images / 255
test_images = test_images / 255

#归一化
train_labels = np.array(pd.get_dummies(train_labels))
test_labels = np.array(pd.get_dummies(test_labels))
#独热编码
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters = 6,kernel_size = (5,5),input_shape=(28,28,1),activation = "relu"))#卷积层,filters为卷积核,默认padding='valid',strides=1
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))#默认strides=None
model.add(tf.keras.layers.Flatten())#展平层
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))#隐藏层,激活方式'sigmoid'
model.add(tf.keras.layers.Dense(10, activation='softmax'))#输出层,10分类
model.summary()
#模型预览
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
#设置优化器,损失函数和记录准确率
history = model.fit(train_images,train_labels,epochs = 10,validation_data=(test_images,test_labels))
#载入训练集,验证集,设置训练轮次
model.evaluate(test_images,test_labels)
#使用测试集进行评估
# 保存模型
model.save('mnist.h5')