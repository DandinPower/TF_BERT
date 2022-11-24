from ..encoder.layer import LinearLayer
from ..encoder.bert import Bert
from ....bert.original.classifier.classifier import Classifier
import tensorflow as tf
from tensorflow.python.framework import ops
import os
from dotenv import load_dotenv
load_dotenv()

@ops.RegisterGradient("BitsQuant")
def _bits_quant_grad(op, grad):
  inputs = op.inputs[0]
  return [grad] 

CLASSIFICATION_TYPES = int(os.getenv('CLASSIFICATION_TYPES'))

class Classifier_Approximate(Classifier):
    def __init__(self, config, parameters):
        super(Classifier_Approximate, self).__init__(config, parameters)
        self.kernel = tf.load_op_library('./models/operations/bits_quant.so')
        self.config = config 
        self.parameters = parameters
        self.bert = Bert(config, self.parameters)
        self.classifier = LinearLayer(config.numHiddens, CLASSIFICATION_TYPES)

    def call(self, inputs):
        output = self.bert(inputs)
        output = self.kernel.bits_quant(self.classifier(output))
        result = self.kernel.bits_quant(tf.nn.softmax(output))
        return result