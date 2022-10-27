import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

class IDataset():
    def __init__(self, datasetPath, max_len, vocab, splitRate):
        self.max_len = max_len
        self.labels = []
        self.vocab = vocab
        self.all_tokens_ids = []
        self.all_segments = []
        self.valid_lens = []
        self.all_datas = []
        self.path = datasetPath
        self.splitRate = splitRate
        self.Preprocess()
    
    #將資料做預處理
    def Preprocess(self):
        texts,self.labels = self.ReadDataset()
        texts = [self.TruncatePairOfTokens(text)for text in texts]
        newTexts,newSegments = [],[]
        for text in texts:
            tokens,segments = self.GetTokensAndSegments(text)
            newTexts.append(tokens)
            newSegments.append(segments)
        self.PadBertInput(newTexts, newSegments)
        self.Merge()

    #讀取dataset 需要實作
    def ReadDataset(self):
        pass

    def GetTokensAndSegments(self,tokensA, tokensB=None):
        tokens = ['<cls>'] + tokensA + ['<sep>']
        # 0 and 1 are marking segment A and B, respectively
        segments = [0] * (len(tokensA) + 2)
        if tokensB is not None:
            tokens += tokensB + ['<sep>']
            segments += [1] * (len(tokensB) + 1)
        return tokens, segments

    #給<CLS>,<SEP>,<SEP>保留位置
    def TruncatePairOfTokens(self, tokens):   
        while len(tokens) > self.max_len - 3:
            tokens.pop()
        return tokens

    #進行padding
    def PadBertInput(self,texts,segments):
        texts = self.vocab[texts]
        for (text,segment) in zip(texts,segments):
            paddingText = np.array(text + [self.vocab['<pad>']] * (self.max_len - len(text)), dtype=np.float32)
            self.all_tokens_ids.append(paddingText)
            self.all_segments.append(np.array(segment + [0] * (self.max_len - len(segment)), dtype=np.float32))
            #valid_lens不包括<pad>
            self.valid_lens.append(np.array(len(text), dtype=np.float32))

    def Merge(self):
        self.all_tokens_ids = tf.constant(self.all_tokens_ids)
        self.all_segments = tf.constant(self.all_segments)
        self.valid_lens = tf.constant(self.valid_lens)
        for i in range(len(self.all_tokens_ids)):
            self.all_datas.append(self.all_tokens_ids[i])

    def GetTrain(self):
        return self.all_tokens_ids[0:self.trainLen], self.all_segments[0:self.trainLen], self.valid_lens[0:self.trainLen], self.labels[0:self.trainLen]

    def GetTest(self):
        return self.all_tokens_ids[self.trainLen:], self.all_segments[self.trainLen:], self.valid_lens[self.trainLen:],self.labels[self.trainLen:]

    def __len__(self):
        return len(self.all_tokens_ids)