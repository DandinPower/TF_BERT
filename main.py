import os
from dotenv import load_dotenv
from preprocess.src.dataLoader import DataLoader
from models.factory.bertFactory import BertFactory
from train.train import Trainer
load_dotenv()

DATASET_TYPE = os.getenv('DATASET_TYPE')
DATASET_PATH = os.getenv('DATASET_PATH')
MAX_LEN = int(os.getenv('MAX_LEN'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
SPLIT_RATES = float(os.getenv('SPLIT_RATES'))
PARAMETER_PATH = os.getenv('PARAMETER_PATH')
BERT_TYPE = os.getenv('BERT_TYPE')

def main():
    trainer = Trainer()
    bertFactory = BertFactory()
    dataLoader = DataLoader()
    dataLoader.SetDataset(DATASET_TYPE, DATASET_PATH, MAX_LEN, SPLIT_RATES, BATCH_SIZE)
    trainDataLoader = dataLoader.GetTrainDataLoader()
    models = bertFactory.GetBert(BERT_TYPE)
    models.LoadParameters()
    trainer.Train(models, trainDataLoader)

if __name__ == "__main__":
    main()