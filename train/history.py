from dotenv import load_dotenv
import os
load_dotenv()

HISTORY_PATH = os.getenv('HISTORY_PATH')

class HistoryRecorder:
    def __init__(self):
        self.history = []

    def Reset(self):
        self.history.clear()

    def AddRecord(self, _acc):
        self.history.append(_acc)

    def WriteHistory(self):
        with open(HISTORY_PATH, "w") as f:
            for item in self.history:
                f.write(f'{item}\n')
        
