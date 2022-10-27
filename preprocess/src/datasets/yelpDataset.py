import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from .IDataset import IDataset
load_dotenv()

# 餐廳評論dataset
class YelpDataset(IDataset):
    def __init__(self, _datasetPath, _max_len, _vocab, _splitRate):
        super().__init__(_datasetPath, _max_len, _vocab, _splitRate)
   
    #讀取dataset
    def ReadDataset(self):
        df = pd.read_csv(self.path)
        labels = []
        texts = []
        for i in range(len(df.Stars.values)):
            text = df.Text.values[i]
            label = df.Stars.values[i]
            if (type(text) != str): continue
            if label >= 4:
                labels.append(1)
            else:
                labels.append(0)
            texts.append(text.strip().lower().split(' '))
        self.trainLen = int(len(df.Text.values) * self.splitRate) 
        return texts,labels
