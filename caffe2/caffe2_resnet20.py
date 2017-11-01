import os
import numpy as np
from caffe2.python import core, model_helper, workspace
from caffe2.proto import caffe2_pb2
import resnet20


data_folder = "/home/r300/git/caffe2/caffe2/python/tutorials/tutorial_data/cifar10"

'''
example for train ResNet-20
'''
workspace.ResetWorkspace()

def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type
    )
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label


def AddTrainingOperators(model, softmax, label):
    # compute the cross entropy
    xent = model.LabelCrossEntropy([softmax, label], "xent")
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    #accuracy = model.Accuracy([softmax, label], "accuracy")
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = model.param_init_net.ConstantFill([], "ITER", 
        shape=[1], value=0, dtype=core.DataType.INT32)
    #ITER = model.Iter("Iter")
    model.Iter(ITER, ITER)
    # set the learning rate schedule
    LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)

        #TODO: add checkpoint every n iters


# Init train_model with NHWC leveldb in data_folder
arg_scope = {"order": "NHWC"}
train_model = model_helper.ModelHelper(name="cifar10_train", arg_scope=arg_scope)

# step 1. Add input data
data, label = AddInput(train_model, batch_size=64,
    db=os.path.join(data_folder, 'cifar10_train'),
    db_type='leveldb'
)

# step 2. define model content
softmax = resnet20.create_resnet20(train_model, data)

# step 3. Add training operators
AddTrainingOperators(train_model, softmax, label)

#train_model.param_init_net.RunAllOnGPU() #gpu_id=0, use_cudnn=True
#train_model.net.RunAllOnGPU()

# In caffe2, all the operators and data you set above are just setting 
# up the workspace. Thus, you need to init the param, create model and then RunNet
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net)

total_iters = 100
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)

print(" ---------- START TRAINING ---------- ")

for i in range(total_iters):
	print("computing " + str(i) + " iters...") 
	workspace.RunNet(train_model.Proto().name)
	#accuracy[i] = workspace.FetchBlob('accuracy')
	loss[i] = workspace.FetchBlob('loss')
	#print("    acc: " + str(accuracy[i]))
	print("    loss: " + str(loss[i]))
	#TODO: to save weights every n iters


#TODO: add test data
