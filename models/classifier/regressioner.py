from ..encoder.layer import LinearLayer
from ..encoder.bert import Bert
import tensorflow as tf

class Regressioner(tf.keras.Model):
    def __init__(self, config, parameters):
        super(Regressioner, self).__init__()
        self.config = config 
        self.parameters = parameters
        self.bert = Bert(config, self.parameters)
        self.linear = LinearLayer(config.numHiddens, config.numHiddens // 4)
        self.linear2 = LinearLayer(config.numHiddens // 4, 1)

    def call(self, inputs):
        output = self.bert(inputs)
        output = self.linear(output)
        output = tf.nn.relu(output)
        output = self.linear2(output)
        result = tf.nn.sigmoid(output)
        return result

    def LoadParameters(self):
        self.bert.LoadParameters()