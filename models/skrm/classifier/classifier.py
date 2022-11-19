from ..encoder.layer import LinearLayer
from ..encoder.bert import Bert
import tensorflow as tf
from ..skrm.skrm import SKRM
from tensorflow.python.framework import ops
import os
from dotenv import load_dotenv
load_dotenv()

@ops.RegisterGradient("CountSkrm")
def _count_skrm_grad(op, grad):
  return [grad] 

CLASSIFICATION_TYPES = int(os.getenv('CLASSIFICATION_TYPES'))

class Classifier_SKRM(tf.keras.Model):
    def __init__(self, config, parameters):
        super(Classifier_SKRM, self).__init__()
        self.config = config 
        self.parameters = parameters
        self.skrm = SKRM()
        self.bert = Bert(config, self.parameters, self.skrm)
        self.classifier = LinearLayer(config.numHiddens, CLASSIFICATION_TYPES)

    def call(self, inputs):
        output = self.bert(inputs)
        output2 = self.classifier(output)
        self.skrm.Count(output, output2)
        result = tf.nn.softmax(output2)
        self.skrm.Count(output2,result)
        return result

    def LoadParameters(self):
        self.bert.LoadParameters()

    def End(self):
        print(self.skrm.GetCount())

    def NewEpoch(self):
        pass