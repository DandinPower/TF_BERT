from .factory import CounterFactory
from .skrmSimulator import SKRM
from dotenv import load_dotenv
import os
load_dotenv()
LOG_TYPE = os.getenv('LOG_TYPE')

class Logger:
    def __init__(self, _id, _blockSize):
        self.id = _id
        self.factory = CounterFactory()
        self.factory.SetBlockSize(_blockSize)
        self.naiveSkrm = SKRM()
        self.skrm = SKRM()
        self.datas = [] #存放多個Counter
    
    def AddNewLog(self, _tensors, _type):
        tempCounter = self.factory.GetCounter(_type)
        tempCounter.SetLog(_tensors)
        self.datas.append(tempCounter)
    
    def AddNewLogByMetadata(self, _metaData):
        tempCounter = self.factory.GetCounterByMetadata(_metaData)
        self.datas.append(tempCounter)

    def ShowLog(self):
        for counter in self.datas:
            print(f'epochs: {self.id}, {counter.ShowLog()}')

    def ShowNaiveResult(self):
        for counter in self.datas:
            if LOG_TYPE == 'All':
                self.naiveSkrm.Add(counter.GetSkrmNaiveRecord())
            elif counter.type == LOG_TYPE:
                self.naiveSkrm.Add(counter.GetSkrmNaiveRecord())
        self.naiveSkrm.Show()

    def ShowImproveResult(self):
        for counter in self.datas:
            if LOG_TYPE == 'All':
                self.skrm.Add(counter.GetSkrmImproveRecord())
            elif counter.type == LOG_TYPE:
                self.skrm.Add(counter.GetSkrmImproveRecord())
        self.skrm.Show()
  
class FullLogger:
    def __init__(self, _blockSize):
        self.counter = 0
        self.epochs = []
        self.blockSize = _blockSize

    def SetNewEpochs(self):
        logger = Logger(self.counter, self.blockSize)
        self.epochs.append(logger)
        self.counter += 1

    def AddNewLog(self, _tensors, _types):
        self.epochs[self.counter - 1].AddNewLog(_tensors, _types)

    def ShowLog(self):
        for logger in self.epochs:
            logger.ShowLog()
    
    def WriteLog(self, _path):
        totalLog = []
        for i in range(len(self.epochs)):
            logger = self.epochs[i].datas
            for j in range(len(logger)):
                counter = logger[j]
                totalLog.append(f'{i};{counter.ShowLog()}\n')
        with open(_path, 'w') as f:
            f.writelines(totalLog)
    
    def ReadLog(self, _path):
        self.epochs.clear()
        with open(_path) as f:
            totalLog = [line.rstrip().split(';') for line in f]
        currentEpoch = int(totalLog[0][0])
        tempLogger = Logger(currentEpoch, self.blockSize)
        for log in totalLog:
            if int(log[0]) != currentEpoch:
                self.epochs.append(tempLogger)
                currentEpoch += 1
                tempLogger = Logger(currentEpoch, self.blockSize)
            tempLogger.AddNewLogByMetadata(log)
        self.epochs.append(tempLogger)

    def ShowNaiveResult(self, _epoch):
        self.epochs[_epoch].ShowNaiveResult()

    def ShowImproveResult(self, _epoch):
        self.epochs[_epoch].ShowImproveResult()