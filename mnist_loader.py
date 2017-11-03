# _*_ coding:utf_8 _*
import cPickle
import gzip
import numpy as np

def load_data():
    # 该函数用来加载数据
    f=gzip.open('../data/mnist.pkl.gz','rb')
    # 数据的存放路径
    training_data,validation_data,test_data=cPickle.load(f)
    f.close
    return(training_data,validation_data,test_data)
def load_data_wrapper():
    # 该函数对数据进行重新封装
    tr_d,va_d,te_d=load_data()
    training_inputs=[np.reshape(x,(784,1)) for x in tr_d[0]]
    # 将输入的数据进行重新封装，将原来的数据封装为784行，1列
    training_results=[vectorized(y) for y in tr_d[1]]
    #将输入数据中对应的输出y 给向量化，例如5，变为[0,0,0,0,1,0,0,0,0,0]
    training_data=zip(training_inputs,training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vectorized(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_results)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    # test_results = [vectorized(y) for  y in te_d[1]]
    test_data = zip(test_inputs, te_d[1])

    return(training_data,validation_data,test_data)

def vectorized(n):
    # 标量进行向量化
    v=np.zeros((10,1))
    v[n]=1.0
    return v
# training_data=load_data_wrapper()[0]
# print type(training_data[0])
# t=0
# print type(training_data)
# for x,y in training_data:
#     t=t+1
#     print y
#     print type(y)
#     print type(x)
#     if t==1:
#         break