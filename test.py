import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from models.logger.logger.skrmSimulator import SKRM

skrm = SKRM()
skrm.shifts = 2343677398834432
skrm.detects = 2343677303574528
#skrm.inserts = 59968
#skrm.removes = 101724
print(skrm.CountLatency())
print(skrm.CountEnergy())
