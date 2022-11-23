from ..encoder.layer import LinearLayer
from ..encoder.bert import Bert
import tensorflow as tf
import os
from dotenv import load_dotenv
load_dotenv()

CLASSIFICATION_TYPES = int(os.getenv('CLASSIFICATION_TYPES'))

class Classifier(tf.keras.Model):
    def __init__(self, config, parameters):
        super(Classifier, self).__init__()
        self.config = config 
        self.parameters = parameters
        self.bert = Bert(config, self.parameters)
        self.classifier = LinearLayer(config.numHiddens, CLASSIFICATION_TYPES)

    def call(self, inputs):
        output = self.bert(inputs)
        output = self.classifier(output)
        result = tf.nn.softmax(output)
        return result

    def LoadParameters(self):
        self.bert.LoadParameters()

    def End(self):
        pass

    def NewEpoch(self):
        pass