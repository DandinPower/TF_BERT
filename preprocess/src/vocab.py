import collections
import json
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
PRETRAIN_DIR_PATH = os.getenv('PRETRAIN_DIR_PATH')

#負責生產出Vocab實體
class VocabFactory:
    def __init__(self):
        pass
    
    # 取得Vocab實體
    def GetVocab(self, _dataDir=PRETRAIN_DIR_PATH):
        data_dir = _dataDir
        vocab = Vocab()
        vocab.idx_to_token = json.load(open(os.path.join(data_dir,
            'vocab.json')))
        vocab.token_to_idx = {token: idx for idx, token in enumerate(
            vocab.idx_to_token)}
        return vocab

# 根據讀入的Vocab表來實作
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def count_corpus(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # Flatten a list of token lists into a list of tokens
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs