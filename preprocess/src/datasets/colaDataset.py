import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from .IDataset import IDataset
load_dotenv()

# Cola dataset
class ColaDataset(IDataset):
    def __init__(self, _datasetPath, _max_len, _vocab, _splitRate):
        super().__init__(_datasetPath, _max_len, _vocab, _splitRate)
   
    #讀取dataset
    def ReadDataset(self):
        df = pd.read_csv(self.path, sep= '\t')
        labels = []
        texts = []
        length = len(df)
        for i in range(length):
            temp = df.iloc[i]
            texts.append(temp[3].strip().lower().split(' '))
            labels.append(temp[1])
        self.trainLen = int(length * self.splitRate) 
        return texts,labels