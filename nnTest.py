from NeuralNetwork import neuralNetwork
import numpy
import matplotlib.pyplot
import csv


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.25

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("seven_segment_dataset_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# go through all records in the training data set

for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    
    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    
    n.train(inputs, targets)
    pass

n.saveWeights("nnWeightsFile")

n.loadWeights("nnWeightsFile.csv")
# load the mnist test data CSV file into a list
test_data_file = open("seven_segment_dataset_test_01.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


all_values = test_data_list[0].split(',')
print(all_values[0])
image_array = numpy.asfarray( all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow( image_array , cmap='Greys', interpolation='None')


result = n.query((numpy.asfarray( all_values[1:]) / 255 * 0.99) + 0.01)

print(result)

