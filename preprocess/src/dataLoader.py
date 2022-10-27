import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from .vocab import VocabFactory
from .datasets.datasetFactory import DatasetFactory
load_dotenv()

# 將dataset轉成batch的形式
class DataLoader():
    def __init__(self, _shuffle=False):
        vocabFactory = VocabFactory()
        self.vocab = vocabFactory.GetVocab()
        self.datasetFactory = DatasetFactory()
        self.shuffle = _shuffle

    # 回傳batchdataset
    def GetBatchDataset(self):
        bufferSize = 1
        if (self.shuffle):
            bufferSize = len(self.labels)
        dataset = tf.data.Dataset.from_tensor_slices((self.tokens, self.segments, self.validLens, self.labels)).batch(self.batch).shuffle(buffer_size = bufferSize)
        return dataset

    # 設定dataset屬性
    def SetDataset(self, _type, _datasetPath, _maxLen, _splitRate, _batchSize):
        print('Loading data....')
        self.dataset = self.datasetFactory.GetDataset(_type, _datasetPath, _maxLen, self.vocab, _splitRate)
        self.batch = _batchSize
    
    # 設定是否要打亂
    def SetShuffle(self, _shuffle):
        self.shuffle = _shuffle

    # 讀取資料集並且生成dataloader
    def GetTrainDataLoader(self):
        train_data = self.dataset.GetTrain()
        self.tokens = train_data[0]
        self.segments = train_data[1]
        self.validLens = train_data[2]
        self.labels = train_data[3]
        self.start = 0
        self.turns = len(self.tokens) // self.batch
        return self.GetBatchDataset()

    # 讀取測試集並且生成dataloader
    def GetTestDataLoader(self):
        test_data = self.dataset.GetTest()
        self.tokens = test_data[0]
        self.segments = test_data[1]
        self.validLens = test_data[2]
        self.labels = test_data[3]
        self.start = 0
        self.turns = len(self.tokens) // self.batch
        return self.GetBatchDataset()

    