import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from models.bert.logger.logger.skrmSimulator import SKRM

skrm = SKRM()
skrm.shifts = 4578845028437
skrm.detects = 2102434975744
skrm.inserts = 75284124902
skrm.removes = 242572890965
print(skrm.Show())
skrm.WriteShow('train/skrm/stsb_approximate_16.csv')


