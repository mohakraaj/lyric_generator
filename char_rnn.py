import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq
from util import TextLoader
import numpy as np
import cPickle
import os

#from reader import Reader 

data_dir ='/tmp/data/shakesphere'
filename = '/tmp/data/shakesphere/input.txt'

keep_probaties = np.arange(0.3,0.8,0.1) # hyper-parameter


class Model():
    
    def __init__(self,infer= False, rnn_size = 256, batch_size=50, seq_length = 50, num_layers=2, 
                 vocab_size = 300, grad_clip = 5, learning_rate = 0.02, 
                 decay_rate = 0.95, num_epocs = 50, display_steps = 24):
        self.batch_size= batch_size
        self.seq_length = seq_length
        self.rnn_size = rnn_size 
        self.num_layers = num_layers
        #vocabulary_size =[500, 2000, 8000, 16000, 32000, 64000] # hyper-parameter
        #self.vocab_size = vocab_size
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate # hyper-parameter
        self.decay_rate = decay_rate
        self.num_epocs = num_epocs
        self.display_steps = display_steps
        self.is_training = True
        self.infer = infer
        if infer:
            self.batch_size = 1
            self.seq_length = 1
        
        self.load_data(data_dir)
        
    def load_data(self, data_dir): 
        self.data_loader = TextLoader(data_dir, self.batch_size, self.seq_length)
        #self.data_loader = Reader(filename, batch_size,seq_length)
        
        #vocabulary_size.append(data_loader.vocab_size)
        self.vocab_size = self.data_loader.vocab_size 
        
        
    def create_model(self):
        
        self.input_data  = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name="input_data")
        self.target_data = tf.placeholder(tf.int32,[self.batch_size, self.seq_length],  name="target_data")

        # define hyper_parameters
        self.keep_prob = tf.Variable(0.3, trainable=False, name='keep_prob')
        self.lr = tf.Variable(0.0, trainable=False, name="lr")
              
        softmax_weights = tf.get_variable("softmax_weights",[self.rnn_size, self.vocab_size])
        softmax_biases = tf.get_variable("softmax_biases", [self.vocab_size])
            
        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size)

        if self.is_training and self.keep_prob < 1:
              lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        
        multilayer_cell = rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
        self.initial_state = multilayer_cell.zero_state(self.batch_size, tf.float32)    
        
        
            
        with tf.device("/cpu:0"):
            # define the embedding matrix for the whole vocabulary
            self.embedding = tf.get_variable("embeddings", [self.vocab_size, self.rnn_size])
            # take the vector representation for each word in the embeddings
            embeds = tf.nn.embedding_lookup(self.embedding, self.input_data)
        
        if self.is_training and self.keep_prob < 1:
            embeds = tf.nn.dropout(embeds, self.keep_prob)
        
        
        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_weights, softmax_biases)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(self.embedding, prev_symbol)
            
        #convert input to a list of seq_length
        inputs = tf.split(1,self.seq_length, embeds)
        
        #after splitting the shape becomes (batch_size,1,rnn_size). We need to modify it to [batch*rnn_size]
        inputs = [ tf.squeeze(input_, [1]) for input_ in inputs]    
    
        output,states= seq2seq.rnn_decoder(inputs,self.initial_state, multilayer_cell, loop_function=loop if self.infer else None, scope='rnnlm')
        
        output = tf.reshape(tf.concat(1, output), [-1, self.rnn_size])
        
        self.logits = tf.nn.xw_plus_b(output, softmax_weights, softmax_biases)
        self.probs = tf.nn.softmax(self.logits, name= "probability")
        
        loss = seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.target_data, [-1])],  [tf.ones([self.batch_size * self.seq_length])], self.vocab_size )
        self.cost = tf.reduce_sum(loss)
        
        self.final_state= states[-1]
        
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),self.grad_clip)
        
        optimizer = tf.train.AdamOptimizer(0.01)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    

    def train_model(self):

        # load data
        self.create_model()

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)        
        saver = tf.train.Saver(tf.all_variables())

        training_cost = float("inf")
        trainingcost_lastepoch = float("inf")
      
        
        with open(os.path.join('chars_vocab.pkl'), 'w') as f:
            cPickle.dump((self.data_loader.chars, self.data_loader.vocab), f)       

        for prob in keep_probaties:
            sess.run(tf.assign(self.keep_prob, prob))
            for e in xrange(self.num_epocs):
                sess.run(tf.assign(self.lr, self.learning_rate * (self.decay_rate ** e)))
                self.data_loader.reset_batch_pointer()
                
                for b in xrange(self.data_loader.num_batches):
                    x, y = self.data_loader.next_batch()
                    #print "X:", x, "Y:", y
                    feed ={self.input_data : x, self.target_data:y }
                    effective_cost,state,_= sess.run([self.cost, self.final_state, self.train_op], feed_dict=feed)  
                    if effective_cost < training_cost:            
                        training_cost = effective_cost
                        
                    if (b % self.display_steps == 0):
                        print " dropout_prob : ", prob,
                        print " Epoch  :", e ,
                        print " and Cost : ", effective_cost            
                
                if training_cost < trainingcost_lastepoch:
                    trainingcost_lastepoch = training_cost
                    saver.save(sess, 'model.ckpt')
                    
                    
    def sample(self, num_of_samples, starting_text = 'The'):
        
        self.create_model()
        
        with open(os.path.join('chars_vocab.pkl')) as f:
            chars, vocab = cPickle.load(f)
        
        def weighted_pick(weights):
            #print "weights :", len(weights)
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))
            
        #store arguments in in a file and load it    
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver(tf.all_variables())
            ckpt = tf.train.get_checkpoint_state('model.ckpt')
            
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)    
            
            #print "Length of chars : ", len(chars)
            #print "vocab", vocab
            ret = starting_text
            char = starting_text[-1]
            state= self.initial_state.eval()
            for i in range(num_of_samples):
                x = np.zeros((1,1))
                x[0,0]= vocab[char]
                feed = {self.input_data :x, self.initial_state:state}        
                [prob, state] = sess.run([self.probs, self.final_state], feed_dict=feed)
                
                p = prob[0]
                # sample = int(np.random.choice(len(p), p=p))
                sample = weighted_pick(p)
                #print "Sample is : ",sample
    
                pred = chars[sample]
                ret += pred
                char = pred
        return ret

if __name__=='__main__':
    model = Model()
    model.train_model()