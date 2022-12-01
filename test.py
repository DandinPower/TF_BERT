import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from models.bert.logger.logger.skrmSimulator import SKRM

cola_original_4_shifts = 6877006449646
cola_original_4_detects = 3153650823168
cola_original_4_inserts = 115591229980
cola_original_4_removes = 372601626862
cola_original_5_shifts = 4682474823352320
cola_original_5_detects = 2341237411676160
cola_original_5_inserts = 0
cola_original_5_removes = 0


cola_approximate_4_shifts = 6504404822784
cola_approximate_4_detects = 3153650823168
cola_approximate_4_inserts = 0
cola_approximate_4_removes = 0
cola_lsm_5_shifts = 2341237621653504
cola_lsm_5_detects = 2341237411676160
cola_lsm_5_inserts = 0
cola_lsm_5_removes = 0

stsb_original_4_shifts = 4577730010111
stsb_original_4_detects = 2102434975744
stsb_original_4_inserts = 74601600519
stsb_original_4_removes = 241457872639
stsb_original_5_shifts = 3121650045550592
stsb_original_5_detects = 1560825022775296
stsb_original_5_inserts = 0
stsb_original_5_removes = 0

stsb_approximate_4_shifts = 4336272137472
stsb_approximate_4_detects = 2102434975744
stsb_approximate_4_inserts = 0
stsb_approximate_4_removes = 0
stsb_lsm_5_shifts = 1560825116098560
stsb_lsm_5_detects = 1560825022775296
stsb_lsm_5_inserts = 0
stsb_lsm_5_removes = 0

skrm = SKRM()
skrm.shifts = cola_approximate_4_shifts + cola_lsm_5_shifts
skrm.detects = cola_approximate_4_detects + cola_lsm_5_detects
skrm.inserts = cola_approximate_4_inserts + cola_lsm_5_inserts
skrm.removes = cola_approximate_4_removes + cola_lsm_5_removes

print(skrm.Show())
skrm.WriteShow('train/skynn/cola_our_16.csv')