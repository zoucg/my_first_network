import lunch_myself
import mnist_loader

my_net=lunch_myself.Network([784,40,20,10])
training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
my_net.SGD(training_data,3.0,20,100,test_data=test_data)