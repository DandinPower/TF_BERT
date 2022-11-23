from .encoder import BERTEncoder
from .block import EncoderBlock
from .layer import LinearLayer
import tensorflow as tf

class Bert(tf.keras.Model):
    def __init__(self, config, parameters, skrm):
        super(Bert, self).__init__()
        self.parameters = parameters
        self.skrm = skrm 
        self.encoder = BERTEncoder(config,parameters,self.skrm)
        self.block1 = EncoderBlock(config,parameters,0,self.skrm,True)
        self.block2 = EncoderBlock(config,parameters,1,self.skrm,True)
        self.hidden = tf.keras.Sequential()
        tempLinearLayer = LinearLayer(config.numHiddens, config.numHiddens)
        tempLinearLayer.set_weights([parameters["hidden.0.weight"],parameters["hidden.0.bias"]])
        self.hidden.add(tempLinearLayer)
        self.hidden.add(tf.keras.layers.Activation('tanh'))

    def call(self, inputs):
        (tokens, segments, valid_lens) = inputs
        embeddingX = self.encoder((tokens,segments))
        X1 = self.block1((embeddingX, valid_lens))
        X2 = self.block2((X1, valid_lens))
        X3 = self.hidden(X2[:, 0, :])
        #encoder已完成
        #block已完成
        self.skrm.Count(X2,X3)
        return X3

    def LoadParameters(self):
        self.encoder.LoadParameters()
        self.block1.LoadParameters()
        self.block2.LoadParameters()