import tensorflow as tf
from progressbar import ProgressBar
from .history import HistoryRecorder
from dotenv import load_dotenv
import os
import time
load_dotenv()
LR = float(os.getenv('LR'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LOSS_TYPE = os.getenv('LOSS_TYPE')

def Classification(_x1, _x2, _x3, y, _model, _loss, _metrics, _optimizer):
        with tf.GradientTape() as tape:
            y_pred = _model((_x1, _x2, _x3))
            loss = _loss(y, y_pred)
            _metrics.update_state(y, y_pred)         
        grads = tape.gradient(loss, _model.variables)
        _optimizer.apply_gradients(grads_and_vars=zip(grads, _model.variables))

class EvaluationFactory:
    def __init__(self):
        pass
    
    def GetLossFunction(self):
        if LOSS_TYPE == 'sparse_categorical_crossentropy':
            return tf.keras.losses.SparseCategoricalCrossentropy()
        elif LOSS_TYPE == 'mean_square_error':
            return tf.keras.losses.MeanSquaredError()

    def GetMetrics(self):
        if LOSS_TYPE == 'sparse_categorical_crossentropy':
            return tf.metrics.SparseCategoricalAccuracy()
        elif LOSS_TYPE == 'mean_square_error':
            return tf.keras.metrics.BinaryAccuracy()

    def GetOptimize(self):
        if LOSS_TYPE == 'sparse_categorical_crossentropy':
            return Classification
        elif LOSS_TYPE == 'mean_square_error':
            return Classification
         
class Trainer:
    def __init__(self):
        self.factory = EvaluationFactory()
        self.history = HistoryRecorder()
        self.loss = self.factory.GetLossFunction()
        self.metrics = self.factory.GetMetrics()
        self.optimize = self.factory.GetOptimize()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    def Train(self, _model, _dataset):
        print('Training...')
        self.model = _model
        self.history.Reset()
        total = len(_dataset)
        for x in range(NUM_EPOCHS):
            startTime = time.time()
            j = 0
            pBar = ProgressBar().start()
            for data in _dataset:
                x1, x2, x3, y = data
                self.optimize(x1, x2, x3, y, self.model, self.loss, self.metrics, self.optimizer)
                pBar.update(int((j / (total - 1)) * 100))
                j += 1
            pBar.finish()
            acc = self.metrics.result().numpy()
            print(f'cost time: {round(time.time() - startTime,3)} sec')
            print(f'epoch:{x} accuracy:{acc}')
            self.history.AddRecord(acc)
            self.metrics.reset_states()
        self.history.WriteHistory()
        