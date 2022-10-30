import collections
import json
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from preprocess.src.dataLoader import DataLoader
from models.classifier.classifier import Classifier
from models.encoder.configs import Config
from train.train import Trainer
from pretrain.load import Parameters, load_variable
load_dotenv()
DATASET_TYPE = os.getenv('DATASET_TYPE')
DATASET_PATH = os.getenv('DATASET_PATH')
MAX_LEN = int(os.getenv('MAX_LEN'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
SPLIT_RATES = float(os.getenv('SPLIT_RATES'))
PARAMETER_PATH = os.getenv('PARAMETER_PATH')

def main():
    trainer = Trainer()
    dataLoader = DataLoader()
    dataLoader.SetDataset(DATASET_TYPE, DATASET_PATH, MAX_LEN, SPLIT_RATES, BATCH_SIZE)
    trainDataLoader = dataLoader.GetTrainDataLoader()
    config = Config()
    parameters = load_variable(PARAMETER_PATH)
    parameters = Parameters(parameters)
    models = Classifier(config, parameters)
    models.LoadParameters()
    trainer.Train(models, trainDataLoader)

if __name__ == "__main__":
    main()