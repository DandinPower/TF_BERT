import tensorflow as tf
import sys

class AddParameter(tf.keras.layers.Layer):
    def __init__(self, nums,hiddens):
        super().__init__()
        self.w = self.add_weight(name='weight',shape=[nums,hiddens], initializer=tf.zeros_initializer())

    def call(self, inputs):
        return inputs + self.w

class BERTEncoder(tf.keras.Model):
    def __init__(self, config, parameters):
        super(BERTEncoder, self).__init__()
        self.kernel = tf.load_op_library('./models/operations/bits_quant.so')
        self.token_embedding = tf.keras.layers.Embedding(config.vocabSize, config.numHiddens,input_length=config.maxLen,weights=[parameters["encoder.token_embedding.weight"]])
        self.segment_embedding = tf.keras.layers.Embedding(2, config.numHiddens,input_length=config.maxLen,weights=[parameters["encoder.segment_embedding.weight"]])
        self.pos_embedding = AddParameter(config.maxLen,config.numHiddens)
        self.config = config
        self.parameters = parameters

    def call(self, inputs):
        (tokens,segments) = inputs
        X = self.kernel.bits_quant(self.token_embedding(tokens))
        X = self.kernel.bits_quant(X + self.kernel.bits_quant(self.segment_embedding(segments)))
        X = self.kernel.bits_quant(self.pos_embedding(X))
        return X

    def LoadParameters(self):
        self.pos_embedding.set_weights(self.parameters["encoder.pos_embedding"])