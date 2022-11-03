import collections
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

#df = pd.read_csv('preprocess/dataset/STS-B/dev.tsv', sep= '\t', error_bad_lines = False)
#print(df.sentence2)

y_pred = [[0.5], [0.5], [0.5]]
y_true = [1, 0, 1]
mse = tf.keras.losses.MeanSquaredError()
loss = mse(y_true, y_pred).numpy()
print(loss)
m = tf.keras.metrics.BinaryAccuracy()
m.update_state(y_true, y_pred)
acc = m.result().numpy()
print(acc)
