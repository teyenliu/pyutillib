# -*- coding: utf-8 -*-

import time
import numpy as np, os, shutil
from   matplotlib import pyplot
from   caffe2.python import core, cnn, net_drawer, workspace, visualize
from caffe2.proto import caffe2_pb2

# Initialization
core.GlobalInit(['caffe2', '--caffe2_log_level=2'])
caffe2_root = '~/caffe2'


data_folder = '/home/r300/git/caffe2/caffe2/python/tutorials/tutorial_data/mnist'

workspace.ResetWorkspace()
device_opts = caffe2_pb2.DeviceOption()
device_opts.device_type = caffe2_pb2.CUDA
device_opts.cuda_gpu_id = 0

#if workspace.has_gpu_support:
#    device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)  # Run on GPU
#else:
#    device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)  # Run on CPU


# Network Input
def AddInput(model, batch_size, db, db_type):
    # Read data and label from DB 
    data_uint8, label = model.TensorProtosDBInput(
        [], ['data_uint8', 'label'], batch_size=batch_size, db=db, db_type=db_type
    )
    
    # uint8 -> float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    
    # [0 255] -> [0 1]
    data = model.Scale(data, data, scale=float(1./256))
    
    # Don't calculate BackProp due to the input data
    data = model.StopGradient(data, data)
    
    return data, label
    
    
# LeNet Model
def AddLeNetModel(model, data):
    # conv -> pool -> conv -> pool -> Relu(fc) -> SoftMax(fc)
    
    # NxCxHxW: 1x1x28x28 -> 20x1x24x24 -> 20x1x12x12
    conv1 = model.Conv(data, 'conv1', 1, 20, 5)
    pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)
    
    # NxCxHxW: 20x1x12x12 -> 50x1x8x8 -> 50x1x4x4
    conv2 = model.Conv(pool1, 'conv2', 20, 50, 5)
    pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)
    
    # NxCxHxW: 800x1x1x1 -> 500x1x1x1
    fc3   = model.FC(pool2, 'fc3', 50 * 4 * 4, 500)
    fc3   = model.Relu(fc3, fc3)
    
    # NxCxHxW: 500x1x1x1 -> 10x1x1x1
    pred  = model.FC(fc3, 'pred', 500, 10)
    softmax = model.Softmax(pred, 'softmax')
    
    return softmax
    
    
# The accuracy calculation
def AddAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], 'accuracy')
    return accuracy
    
    
# Learning
def AddTrainingOperators(model, softmax, label):
    # CrossEntropy
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    
    # Avg loss
    loss = model.AveragedLoss(xent, 'loss')
    
    # Accuracy
    AddAccuracy(model, softmax, label)
    
    # Gradient
    model.AddGradientOperators([loss])
    
    # learning rate（lr = base_lr * (t ^ gamma)）
    ITER = model.Iter('iter')
    LR = model.LearningRate(ITER, 'LR', base_lr=-0.1, policy='step', stepsize=1, gamma=0.999)
    
    # Constant value
    ONE = model.param_init_net.ConstantFill([], 'ONE', shape=[1], value=1.0)
    
    # Update all parameters
    # param = param + param_grad * LR
    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.WeightedSum([param, ONE, param_grad, LR], param)
        
    # every 20 iterations will do checkpoint
    #model.Checkpoint([ITER] + model.params, [], db='mnist_lenet_checkpoint_%05d.leveldb', db_type='leveldb', every=20)
    
    
# Print out the result
def AddBookkeepingOperators(model):
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
        
        

# Here is going to train CNN Model
train_model = cnn.CNNModelHelper(order='NCHW', name='mnist_train', use_cudnn=True, cudnn_exhaustive_search=True)
train_net_def = train_model.net.Proto()
train_net_def.device_option.CopyFrom(device_opts)
train_model.param_init_net.RunAllOnGPU() #gpu_id=0, use_cudnn=True
train_model.net.RunAllOnGPU()

# Read data and label
data, label = AddInput(train_model, batch_size=64, db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'), db_type='leveldb')
# Add LeNet model with training data 
softmax = AddLeNetModel(train_model, data)
# Add training setting
AddTrainingOperators(train_model, softmax, label)
# Add logging
#AddBookkeepingOperators(train_model)

# Here is going to test CN model
test_model = cnn.CNNModelHelper(order='NCHW', name='mnist_test', use_cudnn=True, cudnn_exhaustive_search=True)
test_net_def = test_model.net.Proto()
test_net_def.device_option.CopyFrom(device_opts)
test_model.param_init_net.RunAllOnGPU() #gpu_id=0, use_cudnn=True
test_model.net.RunAllOnGPU()

# Read data and label
data, label = AddInput(test_model, batch_size=100, db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'), db_type='leveldb')
# Add LeNet model with test data
softmax = AddLeNetModel(test_model, data)
# Add the accuracy calculation
AddAccuracy(test_model, softmax, label)

# Here is for deployment
deploy_model = cnn.CNNModelHelper(order='NCHW', name='mnist_deploy', init_params=False)
AddLeNetModel(deploy_model, 'data')


# Begin to do training

# Network initialization
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net)

# pyplot
total_iters = 200
accuracy = np.zeros(total_iters)
loss     = np.zeros(total_iters)

# Training
start = time.time()
for i in xrange(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    
    # plotting charts
    #accuracy[i] = workspace.FetchBlob('accuracy')
    #loss[i]     = workspace.FetchBlob('loss')
    #pyplot.clf()
    #pyplot.plot(accuracy, 'r')
    #pyplot.plot(loss, 'b')
    #pyplot.legend(('loss', 'accuracy'), loc='upper right')
    #pyplot.pause(.01)
print('Training time is Spent: {}'.format((time.time() - start) / total_iters))
    
# Begin to do testing

# Network initialization
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)

# pyplot
total_t_iters = 100
test_accuracy = np.zeros(100)

# testing
start = time.time()
for i in xrange(total_t_iters):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
print('Testing time is Spent: {}'.format((time.time() - start) / total_t_iters))


# plotting the result
#pyplot.plot(test_accuracy, 'r')
#pyplot.title('Acuracy over test batches.')
#print('test_accuracy: %f' % test_accuracy.mean())

