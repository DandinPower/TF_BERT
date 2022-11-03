import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from .IDataset import IDataset
load_dotenv()

# Stsb dataset
class StsbDataset(IDataset):
    def __init__(self, _datasetPath, _max_len, _vocab, _splitRate):
        super().__init__(_datasetPath, _max_len, _vocab, _splitRate)
   
    #讀取dataset
    def ReadDataset(self):
        df = pd.read_csv(self.path, sep= '\t', error_bad_lines = False)
        labels = []
        texts_a = []
        texts_b = []
        length = len(df)
        for i in range(length):
            texts_a.append(df.sentence1[i].strip().lower().split(' '))
            texts_b.append(df.sentence2[i].strip().lower().split(' '))
            labels.append(self.GetTypes(df.score[i]))
        self.trainLen = int(length * self.splitRate) 
        return texts_a, texts_b, labels

    def GetTypes(self, _score):
        if _score <= 5 and _score > 4:
            return 4
        elif _score <= 4 and _score > 3:
            return 3
        elif _score <= 3 and _score > 2:
            return 2
        elif _score <= 2 and _score > 1:
            return 1
        elif _score <= 1 and _score >= 0:
            return 0

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
        