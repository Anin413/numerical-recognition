import tensorflow as tf

import cv2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
#载入数据
new_model = tf.keras.models.load_model('mnist.h5')#调用模型
number = int(input("input a number:"))
img = test_images[number-1]
answer = test_labels[number-1]#设置答案
print("The answer is:\n",test_labels[number-1])
img = cv2.resize(img,(28,28))#处理图片
img = img.reshape(1,28,28,1)
img = img/255
predict = new_model.predict(img)#分类
predict
np.argmax(predict)
result = np.argmax(predict)
print("The predict result is:\n",result)
if result == answer:
    print("The prediction is right.√\n")
else:
    print("The prediction is wrongs.×\n")


