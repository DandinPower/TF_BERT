import sys
from dotenv import load_dotenv
from .yelpDataset import YelpDataset
from .colaDataset import ColaDataset
from .sstDataset import SstDataset
from .qqpDataset import QqpDataset
from .stsbDataset import StsbDataset
load_dotenv()

class DatasetFactory:
    def __init__(self):
        pass 

    # 根據給定的type生成對應的dataset object
    def GetDataset(self, _type, _datasetPath, _maxLen, _vocab, _splitRate):
        if _type == 'YELP':
            return YelpDataset(_datasetPath, _maxLen, _vocab, _splitRate)
        elif _type == 'COLA':
            return ColaDataset(_datasetPath, _maxLen, _vocab, _splitRate)
        elif _type == 'SST':
            return SstDataset(_datasetPath, _maxLen, _vocab, _splitRate)
        elif _type == 'QQP':
            return QqpDataset(_datasetPath, _maxLen, _vocab, _splitRate)
        elif _type == 'STSB':
            return StsbDataset(_datasetPath, _maxLen, _vocab, _splitRate)
        else:
            sys.exit('Dataset Type didn\'t exist')  