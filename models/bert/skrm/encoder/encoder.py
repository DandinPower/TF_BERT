import tensorflow as tf
import sys

class AddParameter(tf.keras.layers.Layer):
    def __init__(self, nums,hiddens):
        super().__init__()
        self.w = self.add_weight(name='weight',shape=[nums,hiddens], initializer=tf.zeros_initializer())

    def call(self, inputs):
        return inputs + self.w

class BERTEncoder(tf.keras.Model):
    def __init__(self, config, parameters, skrms):
        super(BERTEncoder, self).__init__()
        self.skrm = skrms
        self.token_embedding = tf.keras.layers.Embedding(config.vocabSize, config.numHiddens,input_length=config.maxLen,weights=[parameters["encoder.token_embedding.weight"]])
        self.segment_embedding = tf.keras.layers.Embedding(2, config.numHiddens,input_length=config.maxLen,weights=[parameters["encoder.segment_embedding.weight"]])
        self.pos_embedding = AddParameter(config.maxLen,config.numHiddens)
        self.config = config
        self.parameters = parameters

    def call(self, inputs):
        (tokens,segments) = inputs
        output1 = self.token_embedding(tokens)
        output2 = output1 + self.segment_embedding(segments)
        output3 = self.pos_embedding(output2)
        self.skrm.Count(output1, output2)
        self.skrm.Count(output2, output3)
        return output3

    def LoadParameters(self):
        self.pos_embedding.set_weights(self.parameters["encoder.pos_embedding"])