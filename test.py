import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

df = pd.read_csv('preprocess/dataset/SST/train.tsv', sep= '\t')
print(df.sentence)