from .attention import MultiHeadAttention
from .layer import AddNorm,PositionWiseFFN
import tensorflow as tf

class EncoderBlock(tf.keras.Model):
    def __init__(self,config, parameters, index,skrm,use_bias=False):
        super(EncoderBlock, self).__init__()
        self.index = index 
        self.config = config
        self.skrm = skrm
        self.parameters = parameters
        self.attention = MultiHeadAttention(config,parameters,skrm,index,use_bias)
        self.addnorm1 = AddNorm(config.dropout,skrm)
        self.ffn = PositionWiseFFN(config,parameters,skrm,index)
        self.addnorm2 = AddNorm(config.dropout,skrm)

    def call(self, inputs):
        (X, valid_lens) = inputs
        output = self.attention((X, X, X, valid_lens))
        Y = self.addnorm1((X, output))
        output = self.ffn(Y)
        result = self.addnorm2((Y, output))
        return result

    def LoadParameters(self):
        self.ffn.LoadParameters()
        self.attention.LoadParameters()