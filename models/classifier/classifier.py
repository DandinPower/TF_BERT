from ..encoder.layer import LinearLayer
from ..encoder.bert import Bert
import tensorflow as tf

class Classifier(tf.keras.Model):
    def __init__(self, config, parameters):
        super(Classifier, self).__init__()
        self.config = config 
        self.parameters = parameters
        self.bert = Bert(config, self.parameters)
        self.classifier = LinearLayer(config.numHiddens, 2)

    def call(self, inputs):
        output = self.bert(inputs)
        output = self.classifier(output)
        result = tf.nn.softmax(output)
        return result

    def LoadParameters(self):
        self.bert.LoadParameters()