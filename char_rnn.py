import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq
from util import TextLoader
import numpy as np
from reader import Reader 
data_dir ='shakesphere'
filename = '/tmp/data/shakesphere/input.txt'
batch_size= 50
seq_length = 50
rnn_size = 128
num_layers = 2
#vocabulary_size =[500, 2000, 8000, 16000, 32000, 64000] # hyper-parameter
vocab_size = 500
grad_clip = 5
learning_rate = 0.02 # hyper-parameter
decay_rate = 0.95
num_epocs = 50
display_steps = 5
is_training = True
keep_probaties = np.arange(0.3,0.8,0.1) # hyper-parameter
#model

#data_loader = TextLoader(data_dir, batch_size, seq_length)
data_loader = Reader(filename, batch_size,seq_length)
#vocabulary_size.append(data_loader.vocab_size)
vocab_size = data_loader.vocab_size 

# define hyper_parameters
keep_prob = tf.Variable(0.3, trainable=False, name='keep_prob')
lr = tf.Variable(0.0, trainable=False, name="lr")

softmax_weights = tf.get_variable("softmax_weights",[rnn_size, vocab_size])
softmax_biases = tf.get_variable("softmax_biases", [vocab_size])


lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
if is_training and keep_prob < 1:
      lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

multilayer_cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
initial_state = multilayer_cell.zero_state(batch_size, tf.float32)    


input_data  = tf.placeholder(tf.int32, [batch_size, seq_length])
target_data = tf.placeholder(tf.int32,[batch_size, seq_length])



with tf.device("/cpu:0"):
    # define the embedding matrix for the whole vocabulary
    embedding = tf.get_variable("embeddings", [vocab_size, rnn_size])
    # take the vector representation for each word in the embeddings
    embeds = tf.nn.embedding_lookup(embedding, input_data)

if is_training and keep_prob < 1:
    embeds = tf.nn.dropout(embeds, keep_prob)

#convert input to a list of seq_length
inputs = tf.split(1,seq_length, embeds)

#after splitting the shape becomes (batch_size,1,rnn_size). We need to modify it to [batch*rnn_size]
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs]    


    

output,states= seq2seq.rnn_decoder(inputs,initial_state, multilayer_cell)

output = tf.reshape(tf.concat(1, output), [-1, rnn_size])

logits = tf.nn.xw_plus_b(output, softmax_weights, softmax_biases)
probs = tf.nn.softmax(logits, name= "probability")

loss = seq2seq.sequence_loss_by_example([logits], [tf.reshape(target_data, [-1])],  [tf.ones([batch_size * seq_length])], vocab_size )
cost = tf.reduce_sum(loss)

final_state= states[-1]

tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),grad_clip)

optimizer = tf.train.AdamOptimizer(0.01)
train_op = optimizer.apply_gradients(zip(grads, tvars))


init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)


saver = tf.train.Saver(tf.all_variables())

training_cost = float("inf")
trainingcost_lastepoch = float("inf")

for prob in keep_probaties:
    sess.run(tf.assign(keep_prob, prob))
    for e in xrange(num_epocs):
        sess.run(tf.assign(lr, learning_rate * (decay_rate ** e)))
        data_loader.reset_batch_pointer()
        
        for b in xrange(data_loader.num_batches):
            x, y = data_loader.next_batch()
            feed ={input_data : x, target_data:y }
            effective_cost,state,_= sess.run([cost, final_state, train_op], feed_dict=feed)  
            if effective_cost < training_cost:            
                training_cost = effective_cost
                
            if (b % display_steps == 0):
                print " dropout_prob : ", prob,
                print " Epoch  :", e ,
                print " and Cost : ", effective_cost            
        
        if training_cost < trainingcost_lastepoch:
            trainingcost_lastepoch = training_cost
            saver.save(sess, 'model.ckpt')

