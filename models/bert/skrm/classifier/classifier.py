from ..encoder.layer import LinearLayer
from ..encoder.bert import Bert
from ....bert.original.classifier.classifier import Classifier
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

class Classifier_SKRM(Classifier):
    def __init__(self, config, parameters):
        super(Classifier_SKRM, self).__init__(config, parameters)
        self.config = config 
        self.parameters = parameters
        self.skrm = SKRM()
        self.bert = Bert(config, self.parameters, self.skrm)
        self.classifier = LinearLayer(config.numHiddens, CLASSIFICATION_TYPES)
        self.count = 0

    def call(self, inputs):
        self.count += 1
        if (self.count % 1000 == 0):
            self.Update()
        output = self.bert(inputs)
        output2 = self.classifier(output)
        self.skrm.Count(output, output2)
        result = tf.nn.softmax(output2)
        self.skrm.Count(output2,result)
        return result

    def Update(self):
        self.skrm.Store()

    def End(self):
        print(self.skrm.store)