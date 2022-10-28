import collections
import json
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from preprocess.src.dataLoader import DataLoader
load_dotenv()
DATASET_TYPE = os.getenv('DATASET_TYPE')
DATASET_PATH = os.getenv('DATASET_PATH')
MAX_LEN = int(os.getenv('MAX_LEN'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
SPLIT_RATES = float(os.getenv('SPLIT_RATES'))

def main():
    dataLoader = DataLoader()
    dataLoader.SetDataset(DATASET_TYPE, DATASET_PATH, MAX_LEN, SPLIT_RATES, BATCH_SIZE)
    trainDataLoader = dataLoader.GetTrainDataLoader()
    x = 0
    for a in trainDataLoader:
        if x == 0:
            print(a)
        x += 1

if __name__ == "__main__":
    main()