import input_data
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn,rnn_cell

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

learning_rate = 0.001
training_iters = 300000
batch_size = 64
display_step = 100

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input], name= "x_input")
y = tf.placeholder("float",[None, n_classes], name="y_truth")
istate = tf.placeholder("float", [None, 2 * n_hidden], name ="lstm_internal_states") #state and cell needs to be created for lstm

weights = {
'hidden' : tf.Variable(tf.random_normal([n_input, n_hidden]), name ="weights_hidden"),
'out' : tf.Variable(tf.random_normal([n_hidden, n_classes]), name="weights_out")
}



biases = {
'hidden' : tf.Variable(tf.random_normal([n_hidden]), name="biases_hidden"),
'out' : tf.Variable(tf.random_normal([n_classes]), name="biases_out")
}

# x [no_images, nsteps, n_inputs]
x = tf.transpose(x, [1,0,2])  # no idea why we do transpose

x = tf.reshape(x,[-1, n_input]) # convert to [_ , input ] shape which is supposed to be format 

x_xh = tf.matmul(x, weights['hidden']) + biases['hidden']

x_split_list = tf.split(0,n_steps, x_xh)

lstm_cell = rnn_cell.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0)

lstm_output, states = rnn.rnn(lstm_cell, x_split_list, initial_state=istate)

# lstm_output contains all internals LSTM outputs excepts last one

output = lstm_output[-1]
with tf.name_scope("predicted_output") as scope:
    y_ = tf.matmul(output, weights['out']) + biases['out']


cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# evaluvate the model
correct_pred = tf.equal(tf.argmax(y,1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    
    step =1 
    
    while step * batch_size < training_iters:
        
        step = step + 1
        batch_xs, batch_y = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        
        feed = {x:batch_xs, y : batch_y,istate: np.zeros((batch_size, 2*n_hidden))}        
        sess.run(optimizer, feed_dict=feed)
        
        if step % display_step ==0:
            
            acc = sess.run(accuracy, feed_dict=feed)
            loss = sess.run(cost, feed_dict=feed)
            print ("iteration step ", step," has cost: ",loss," and accuracy: ",acc)
    
    test_len = 256
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]    
    acc= sess.run(accuracy,feed_dict={x: test_data, y: test_label, istate: np.zeros([test_len,2*n_hidden])})
    
    print "Classification accuracy is:", acc