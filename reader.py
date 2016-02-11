from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np

from tensorflow.python.platform import gfile

class Reader():
    
    def __init__(self, filename, seq_length, batch_size):
        self.filename = filename
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.word_to_id = self._build_vocab()
        self.file_to_id = self._file_to_word_ids()
        self.vocab_size = len(self.word_to_id)
        self.num_batches = int(len(self.file_to_id) / (self.batch_size * self.seq_length))
        self.pointer = 0
        self.create_batches()
        self.reset_batch_pointer()
        
    def _read_words(self):
      with gfile.GFile(self.filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

    def _build_vocab(self):
      data = self._read_words()
    
      counter = collections.Counter(data)
      count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    
      words, _ = list(zip(*count_pairs))
      word_to_id = dict(zip(words, range(len(words))))
    
      return word_to_id
    
    
    def _file_to_word_ids(self):
      data = self._read_words()
      data= [self.word_to_id[word] for word in data]
      return data [ : len(data) - len(data) %self.batch_size ]
      
    def create_batches(self):
        xdata = np.array(self.file_to_id)
        xdata = xdata[:self.num_batches * self.batch_size * self.seq_length]
        ydata = np.copy(xdata)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

filename = '/tmp/data/shakesphere/input.txt'
batch_size= 50
seq_length = 50

data_loader = Reader(filename, batch_size,seq_length)