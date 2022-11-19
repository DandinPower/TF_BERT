from .encoder import BERTEncoder
from .block import EncoderBlock
from .layer import LinearLayer
import tensorflow as tf

class Bert(tf.keras.Model):
    def __init__(self, config, parameters, skrm):
        super(Bert, self).__init__()
        self.parameters = parameters
        self.skrm = skrm 
        self.encoder = BERTEncoder(config,parameters)
        self.block1 = EncoderBlock(config,parameters,0,True)
        self.block2 = EncoderBlock(config,parameters,1,True)
        self.hidden = tf.keras.Sequential()
        tempLinearLayer = LinearLayer(config.numHiddens, config.numHiddens)
        tempLinearLayer.set_weights([parameters["hidden.0.weight"],parameters["hidden.0.bias"]])
        self.hidden.add(tempLinearLayer)
        self.hidden.add(tf.keras.layers.Activation('tanh'))

    def call(self, inputs):
        (tokens, segments, valid_lens) = inputs
        embeddingX = self.encoder((tokens,segments))
        X = self.block1((embeddingX, valid_lens))
        X = self.block2((X, valid_lens))
        X = self.hidden(X[:, 0, :])
        return X

    def LoadParameters(self):
        self.encoder.LoadParameters()
        self.block1.LoadParameters()
        self.block2.LoadParameters()