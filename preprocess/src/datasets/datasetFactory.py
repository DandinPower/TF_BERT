import collections
import json
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from .yelpDataset import YelpDataset
load_dotenv()

class DatasetFactory:
    def __init__(self):
        pass 

    # 根據給定的type生成對應的dataset object
    def GetDataset(self, _type, _datasetPath, _maxLen, _vocab, _splitRate):
        if _type == 'YELP':
            return YelpDataset(_datasetPath, _maxLen, _vocab, _splitRate)
        else:
            sys.exit('Dataset Type didn\'t exist')  