import tensorflow as tf
import math
from .layer import LinearLayer
import time

class DotProductAttention(tf.keras.Model):
    def __init__(self, dropout,config):
        super(DotProductAttention, self).__init__()
        self.kernel = tf.load_op_library('./models/operations/bits_quant.so')
        self.config = config
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.first = self.config.batchSize * self.config.numHeads
        self.second = self.config.maxLen
        self.third = int(self.config.numHiddens//self.config.numHeads)

    def call(self, inputs):
        (queries, keys, values, valid_lens) = inputs
        if (queries.shape[0]==None):
            queries = tf.reshape(queries,[self.first,self.second,self.third])
            keys = tf.reshape(keys,[self.first,self.second,self.third])
            values = tf.reshape(values,[self.first,self.second,self.third])
        if (valid_lens.shape[0] == None):
            valid_lens = tf.reshape(valid_lens,[self.first,])
        d = queries.shape[-1]
        keys = tf.transpose(keys,[0,2,1])
        scores = self.kernel.bits_quant(self.kernel.bits_quant(tf.matmul(queries, keys)) / self.kernel.bits_quant(math.sqrt(d)))
        self.attention_weights = self.masked_softmax((scores,valid_lens))
        result = tf.matmul(self.kernel.bits_quant(self.dropout(self.attention_weights)), values)
        return self.kernel.bits_quant(result)

    def sequence_mask(self,X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range((maxlen),dtype=tf.float32)[None, :] < valid_len[:, None]
        X = tf.where(mask,X,value)
        return self.kernel.bits_quant(X)

    def masked_softmax(self,inputs):
        (X,valid_lens) = inputs
        if valid_lens is None:
            return tf.nn.softmax(X, axis=-1)
        else:
            shape = X.shape
            if len(valid_lens.get_shape()) == 1:
                valid_lens = tf.repeat(valid_lens, repeats=shape[1])
            else:
                valid_lens = tf.reshape(valid_lens,[-1])
            X = self.sequence_mask(tf.reshape(X,[-1, shape[-1]]), valid_lens,
                                value=-1e6)
            X = tf.reshape(X,shape)
            result = tf.nn.softmax(X, axis=-1)
            return self.kernel.bits_quant(result)


class MultiHeadAttention(tf.keras.Model):
    def __init__(self,config,parameters,index,bias=False):
        super(MultiHeadAttention, self).__init__()
        self.kernel = tf.load_op_library('./models/operations/bits_quant.so')
        self.num_heads = config.numHeads
        self.config = config 
        self.parameters = parameters 
        self.index = index 
        self.attention = DotProductAttention(config.dropout,config)
        self.W_q = LinearLayer(config.numHiddens, config.numHiddens)
        self.W_k = LinearLayer(config.numHiddens, config.numHiddens)
        self.W_v = LinearLayer(config.numHiddens, config.numHiddens)
        self.W_o = LinearLayer(config.numHiddens, config.numHiddens)

    def call(self, inputs):
        (queries, keys, values, valid_lens) = inputs
        queries = self.kernel.bits_quant(self.W_q(queries))
        queries = self.transpose_qkv((queries, self.num_heads))
        keys = self.transpose_qkv((self.W_k(keys), self.num_heads))
        values = self.transpose_qkv((self.W_v(values), self.num_heads))
        if (valid_lens.shape[0] == None):
            valid_lens = tf.reshape(valid_lens,[self.config.batchSize,])
        valid_lens = tf.repeat(valid_lens,repeats = self.num_heads,axis=0)
        output = self.attention((queries, keys, values ,valid_lens))
        output_concat = self.transpose_output((output, self.num_heads))
        result = self.W_o(output_concat)
        return self.kernel.bits_quant(result)

    def transpose_qkv(self,inputs):
        (X, num_heads) = inputs
        X = tf.reshape(X,[self.config.batchSize, X.shape[1], num_heads, (self.config.numHiddens//num_heads)])
        X = tf.transpose(X , perm = [0,2,1,3])
        return tf.reshape(X,[-1, X.shape[2], X.shape[3]])

    def transpose_output(self,inputs):
        (X, num_heads) = inputs
        X = tf.reshape(X,[self.config.batchSize, num_heads, X.shape[1], X.shape[2]])
        X = tf.transpose(X , perm = [0,2,1,3])
        return tf.reshape(X,[X.shape[0], X.shape[1], -1])

    def LoadParameters(self):
        self.W_q.set_weights([self.parameters[f"encoder.blks.{self.index}.attention.W_q.weight"],self.parameters[f"encoder.blks.{self.index}.attention.W_q.bias"]])
        self.W_k.set_weights([self.parameters[f"encoder.blks.{self.index}.attention.W_k.weight"],self.parameters[f"encoder.blks.{self.index}.attention.W_k.bias"]])
        self.W_v.set_weights([self.parameters[f"encoder.blks.{self.index}.attention.W_v.weight"],self.parameters[f"encoder.blks.{self.index}.attention.W_v.bias"]])
        self.W_o.set_weights([self.parameters[f"encoder.blks.{self.index}.attention.W_o.weight"],self.parameters[f"encoder.blks.{self.index}.attention.W_o.bias"]])
