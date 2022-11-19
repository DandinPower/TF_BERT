import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from models.logger.logger.skrmSimulator import SKRM

skrm = SKRM()
skrm.shifts = 477815111680
print(skrm.CountLatency())
print(skrm.CountEnergy())
