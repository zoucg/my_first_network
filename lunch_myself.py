# _*_ coding: utf_8 _*
# 程序中如果出现中文字符，需要加第一行注释，否则会编译出错
import numpy as np
import random

class Network(object):
    def __init__(self,sizes):
        # 初始化
        self.length=len(sizes)
        self.bias=[np.random.randn(x,1) for x in sizes[1:]]
        # 产生一系列数组，数组的个数为层数减一，数组的结构为：行为该层的神经元个数，列为一

        self.weights=[np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        # 同理产生一系列的数组，数组的个数为层数减一，数组的结构为：行为该层的神经元个数，列为前一层的神经元个数

    def  feedforward(self,a):
        # 前向传递函数，该函数的作用为:将输入前向传递，直到最后一层进行输出
        for w,b in zip(self.weights,self.bias):
            # 每次取一个，权重矩阵和偏差矩阵
            z=np.dot(w,a)+b
            a=sigmoid(z)
            # 返回最后一层的激活值
        return a

    def SGD(self,training_data,eta,epoch,mini_batch_size,test_data=None):
        # 随机梯度下降算法，stotistic gradient descent,
        # training_data 为训练数据；
        # eta 为学习速率；
        # epoch，为测试的轮数，也是学习的轮数；
        # mini_batch_size 为训练数据分组后的每一组含有的数据个数
        #  test_data为测试数据


        if test_data:
            n_test=len(test_data)
            # 测试的数据个数
        n=len(training_data)
        # 训练的数据个数

        for j in xrange(epoch):
            random.shuffle(training_data)
            mini_batchs=[training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            # 训练数据分组，每一组mini_batch_size 个数据
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print"test{0}：{1}/{2}".format(j,self.evaluate(test_data),n_test)
                # print self.bias[1]
            else :
                print "test{0}complete".format(j)


    def update_mini_batch(self,mini_batch,eta):
        # 采用梯度下降法更新权重和偏差
        nabla_b=[np.zeros(b.shape) for b in self.bias]
        #初始化偏差数组
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        # 初始化权重数组

        for x,y in mini_batch:
            # 对每一组的梯度进行累加
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb ,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw ,dnw in zip(nabla_w, delta_nabla_w)]
        self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.bias=[b-(eta/len(mini_batch))*nb for b,nb in  zip(self.bias,nabla_b)]

    def backprop(self,x,y):
        # 利用反向传播算法求梯度
        nabla_b=[np.zeros(b.shape) for b in self.bias]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation=x
        activations=[x]
        zs=[]

        for b ,w in zip(self.bias ,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
        delta=self.cost_derivation(activations[-1],y)*sigmoid_prime(zs[-1])
        # 此处delta的定义为 cost_function 对最后一层b的偏导
        # 采用的是BP(1)
        nabla_b[-1] = delta
        # 和BP(3)公式
        nabla_w[-1] =np.dot(delta,activations[-2].transpose())
        # 由于此处nabla_w是一个矩阵，必须注意行列的对应，行对应delta 的列，
        # 列对应上一层activation的的列
        # 此处用到的是BP(2)和BP(4)公式


        for j in xrange(2,self.length):
            # 轮换应用BP(1)BP(2)BP(3)BP(4),求出每一层的梯度
            z=zs[-j]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-j+1].transpose(),delta)*sp
            nabla_b[-j]=delta
            nabla_w[-j]=np.dot(delta,activations[-j-1].transpose())
            # 此处delta 和activation 的顺序不能相反
        return(nabla_b,nabla_w)


    def cost_derivation(self,output_activation,y):
        return (output_activation-y)

    def evaluate(self,test_data):
        # 该函数的作用是统计用来测试的数据有多少识别对了
        test_results=[(np.argmax(self.feedforward(x)),y)for x ,y in test_data]
        return sum(int(x==y) for (x,y) in test_results)

def sigmoid(z):
        return 1.0/(1+np.exp(-z))

def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))
