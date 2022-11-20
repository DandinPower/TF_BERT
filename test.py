import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from models.logger.logger.skrmSimulator import SKRM
from models.logger.logger.logger import FullLogger

'''
block = [[4, 4], [4, 4]]
logger = FullLogger(block)
logger.ReadLog('train/history/log/lstm_operation.txt')
logger.ShowNaiveResult(0)
logger.ShowImproveResult(0)
'''


cola_original_shifts= 2311456 + 7031031660281856
cola_our_shifts = 2311456 + 3515516040118272
cola_original_detects = 1093632 + 3515515830140928
cola_our_detects = 1093632 + 3515515830140928
cola_original_inserts = 38364
cola_our_inserts = 0
cola_original_removes = 55840
cola_our_removes = 0

stsb_original_shifts= 3861084 + 4687354603503616
stsb_our_shifts = 3861084 + 2343677395075072
stsb_original_detects = 1822720 + 2343677301751808
stsb_our_detects = 1822720 + 2343677301751808
stsb_original_inserts = 59968
stsb_our_inserts = 0
stsb_original_removes = 101724
stsb_our_removes = 0

lstm_original_shifts= 67844092 + 219167360
lstm_our_shifts = 67844092 + 118691200
lstm_original_detects = 0 + 109583680
lstm_our_detects = 0 + 109583680
lstm_original_inserts = 717535
lstm_our_inserts = 0
lstm_original_removes = 661504
lstm_our_removes = 0

skrm = SKRM()
skrm.shifts = lstm_our_shifts
skrm.detects = lstm_our_detects
skrm.inserts = lstm_our_inserts
skrm.removes = lstm_our_removes
print(skrm.Show())