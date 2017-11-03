# my_first_network
这是我的第一个关于深度学习的小项目。第一次使用GitHub发布代码，算是练习一下使用GitHub，以后会陆陆续续发布各种学习笔记。
这个小项目是看了几遍Miceal Nielsen 的电子书后一二章之后的一个复现。发现自己写的时候出现了各种错误。基本断断续续写完用了两三天,该代码包含详细的中文注释。

该项目一共包含三个python文件，一个为lunch_myself.py，一个mnist_loader.py,start.py.

1.其中lunch_myself 这是神经网络模型的主要构成部分，包含神经网络的结构，weights、bias 的初始化，随机梯度下降算法更新weights和bias，反向传播算法求偏导等函数。
2.mnist_loader.py 包含对数据的加载，重新封装成不同的数组。注意加载数据时候的路径问题，加载的数据应放在项目文件的外层。

3.start.py 为启动代码,创建一个network 实例，并调用SGD 函数来训练，并利用test_data 来测试。
