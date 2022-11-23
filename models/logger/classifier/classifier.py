from ..encoder.layer import LinearLayer
from ..encoder.bert import Bert
from ..logger.logger import FullLogger
import tensorflow as tf
import os
from dotenv import load_dotenv
load_dotenv()

CLASSIFICATION_TYPES = int(os.getenv('CLASSIFICATION_TYPES'))
BLOCK_SIZE_A_ROWS = int(os.getenv('BLOCK_SIZE_A_ROWS'))
BLOCK_SIZE_A_COLS = int(os.getenv('BLOCK_SIZE_A_COLS'))
BLOCK_SIZE_B_COLS = int(os.getenv('BLOCK_SIZE_B_COLS'))
LOG_PATH = os.getenv('LOG_PATH')

blockSize = [[BLOCK_SIZE_A_ROWS, BLOCK_SIZE_A_COLS], [BLOCK_SIZE_A_COLS, BLOCK_SIZE_B_COLS]]

class Classifier_Logger(tf.keras.Model):
    def __init__(self, config, parameters):
        super(Classifier_Logger, self).__init__()
        self.logger = FullLogger(blockSize)
        self.config = config 
        self.parameters = parameters
        self.bert = Bert(config, self.parameters, self.logger)
        self.classifier = LinearLayer(config.numHiddens, CLASSIFICATION_TYPES)

    def call(self, inputs):
        output = self.bert(inputs)
        shape1 = [output.shape[0], output.shape[1]]
        output = self.classifier(output)
        shape2 = [self.config.numHiddens, CLASSIFICATION_TYPES]
        result = tf.nn.softmax(output)
        self.logger.AddNewLog([shape1, shape2], "matmul")
        return result

    def LoadParameters(self):
        self.bert.LoadParameters()

    def NewEpoch(self):
        self.logger.SetNewEpochs()

    def Update(self):
        pass
    
    def End(self):
        self.logger.WriteLog(LOG_PATH)
        self.logger.ShowNaiveResult(0)
        self.logger.ShowImproveResult(0)