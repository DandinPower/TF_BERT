import tensorflow as tf
from progressbar import ProgressBar
from dotenv import load_dotenv
import os
import time
load_dotenv()
LR = float(os.getenv('LR'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
LOSS_TYPE = os.getenv('LOSS_TYPE')

class EvaluationFactory:
    def __init__(self):
        pass
    
    def GetLossFunction(self):
        if LOSS_TYPE == 'sparse_categorical_crossentropy':
            return tf.keras.losses.SparseCategoricalCrossentropy()

    def GetMetrics(self):
        if LOSS_TYPE == 'sparse_categorical_crossentropy':
            return tf.metrics.SparseCategoricalAccuracy()

class Trainer:
    def __init__(self):
        self.factory = EvaluationFactory()
        self.history = []
        self.loss = self.factory.GetLossFunction()
        self.metrics = self.factory.GetMetrics()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    def WriteHistory(self, _path):
        with open(_path, 'w') as f:
            for item in self.history:
                f.write(f'{item}\n')

    def Train(self, _model, _dataset):
        print('Training...')
        self.model = _model
        self.history.clear()
        total = len(_dataset)
        for x in range(NUM_EPOCHS):
            startTime = time.time()
            j = 0
            pBar = ProgressBar().start()
            for data in _dataset:
                x1, x2, x3, y = data
                with tf.GradientTape() as tape:
                    y_pred = self.model((x1, x2, x3))
                    loss = self.loss(y, y_pred)
                    self.metrics.update_state(y, y_pred)
                    loss = tf.reduce_mean(loss)          
                grads = tape.gradient(loss, self.model.variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))
                pBar.update(int((j / (total - 1)) * 100))
                j += 1
            pBar.finish()
            acc = self.metrics.result().numpy()
            print(f'cost time: {round(time.time() - startTime,3)} sec')
            print(f'epoch:{x} accuracy:{acc}')
            self.history.append(acc)
            self.metrics.reset_states()
        