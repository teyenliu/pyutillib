import tensorflow as tf

""" 
x = tf.Variable(initial_value=3.0)
add = tf.add(x, 1)
y = tf.square(add)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(y)
"""

# This register function is to define a new gradient op.
@tf.RegisterGradient("DannySigmoidGrad")
def grad_danny_sigmoid(op, input_data):
    return input_data * 1.1

@tf.RegisterGradient("DannyCustomGrad")
def grad_danny_custom(op, input_data):
    return input_data / 1.1

G = tf.get_default_graph()

# Network Parameters
n_input = 4  # ESL data input
n_final = 2

# tf Graph input (only pictures)
X = tf.placeholder("float32", [None, n_input], "ESL_input_data")
Y = tf.placeholder("float32", [None, n_final], "ESL_output_data")

# hidden layer settings
n_hidden_1 = 10
n_hidden_2 = 10


weights = {
    'weights_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'weights_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'weights_output': tf.Variable(tf.truncated_normal([n_hidden_2, n_final],)),
}

biases = {
    'biases_h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'biases_h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'biases_output': tf.Variable(tf.random_normal([n_final])),

}

# using gradient_override_map, we can replace the original sigmoid
# gradient op to our defined DannySigmoid op
with G.gradient_override_map({"Sigmoid": "DannySigmoidGrad"}):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['weights_h1']),
                               biases['biases_h1']))

with G.gradient_override_map({"Identity": "DannyCustomGrad"}):
    layer_1_swapout = tf.identity(layer_1, name = "L1_SwapOut")
    layer_1_swapin = tf.identity(layer_1_swapout, name = "L1_SwapIn")

layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['weights_h2']),
                               biases['biases_h2']))

layer_final = tf.add(tf.matmul(layer_2, weights['weights_output']),
                               biases['biases_output'])


# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.square(layer_final - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Modify the graph
from tensorflow.contrib import graph_editor as ge

sess = tf.Session()
all_ops = sess.graph.get_operations()

l1_swap_in = sess.graph.get_operation_by_name("L1_SwapIn")
matmul_1_grad = sess.graph.get_operation_by_name("gradients/MatMul_1_grad/MatMul_1")
print(ge.sgv(matmul_1_grad))
print(ge.sgv(l1_swap_in))

ret = ge.connect(ge.sgv(l1_swap_in), ge.sgv(matmul_1_grad).remap_inputs([0]))
sess.close()

# Generate graph visualization
graph = tf.get_default_graph()
writer = tf.summary.FileWriter("./simple_graph_events")
writer.add_graph(graph=graph)

