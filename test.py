import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

history = [1, 0, 1]
with open("a_file.txt", "w") as f:
    for item in history:
        f.write(f'{item}\n')
