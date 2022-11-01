import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from .IDataset import IDataset
load_dotenv()

# qqp dataset
class QqpDataset(IDataset):
    def __init__(self, _datasetPath, _max_len, _vocab, _splitRate):
        super().__init__(_datasetPath, _max_len, _vocab, _splitRate)
   
    #讀取dataset
    def ReadDataset(self):
        df = pd.read_csv(self.path, sep= '\t')
        labels = []
        texts_a = []
        texts_b = []
        length = len(df)
        for i in range(length):
            texts_a.append(df.question1[i].strip().lower().split(' '))
            texts_b.append(df.question2[i].strip().lower().split(' '))
            labels.append(df.is_duplicate[i])
        self.trainLen = int(length * self.splitRate) 
        return texts_a, texts_b, labels

    #將資料做預處理
    def Preprocess(self):
        texts_a, texts_b, self.labels = self.ReadDataset()
        texts_a = [self.TruncatePairOfTokens(text)for text in texts_a]
        texts_b = [self.TruncatePairOfTokens(text)for text in texts_b]
        newTexts,newSegments = [],[]
        for text_a, text_b in zip(texts_a, texts_b):
            tokens,segments = self.GetTokensAndSegments(text_a, text_b)
            newTexts.append(tokens)
            newSegments.append(segments)
        self.PadBertInput(newTexts, newSegments)
        self.Merge()