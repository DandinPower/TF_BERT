import os
import sys
from dotenv import load_dotenv
from models.original.classifier.classifier import Classifier
from models.original.classifier.regressioner import Regressioner
from models.original.encoder.configs import Config
from models.approximate.classifier.classifier import Classifier_Approximate
from models.logger.classifier.classifier import Classifier_Logger
from pretrain.load import Parameters, load_variable
load_dotenv()
PARAMETER_PATH = os.getenv('PARAMETER_PATH')

class BertFactory:
    def __init__(self):
        self.config = Config()
        parameters = load_variable(PARAMETER_PATH)
        self.parameters = Parameters(parameters)

    def GetBert(self, _type):
        if _type == 'Original':
            return Classifier(self.config, self.parameters)
        elif _type == 'Approximate':
            return Classifier_Approximate(self.config, self.parameters)
        elif _type == 'Logger':
            return Classifier_Logger(self.config, self.parameters)
        else:
            sys.exit('Model Type didn\'t exist')  