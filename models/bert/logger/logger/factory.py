from .counter import Counter, MatmulCounter, TransposeCounter

class CounterFactory:
    def __init__(self):
        self.matmulNums = 0
        self.transposeNums = 0

    def SetBlockSize(self, _blockSize):
        self.blockSize = _blockSize

    def GetCounter(self, _type):
        if _type == "matmul":
            newCounter = MatmulCounter(self.matmulNums)
            newCounter.SetBlockSize(self.blockSize)
            self.matmulNums += 1
            return newCounter
        elif _type == "transpose":
            newCounter = TransposeCounter(self.transposeNums)
            self.transposeNums += 1
            return newCounter

    def GetCounterByMetadata(self, _metaData):
        counterType = _metaData[1]
        counterId = int(_metaData[2])
        if counterType == "matmul":
            newCounter = MatmulCounter(counterId)
            newCounter.SetBlockSize(self.blockSize)
            tensors = [eval(_metaData[3]), eval(_metaData[4])]
            newCounter.SetLog(tensors)     
        elif counterType == "transpose":
            newCounter = TransposeCounter(counterId)
            tensors = [eval(_metaData[3])]
            newCounter.SetLog(tensors)
        return newCounter

class SKRM:
    def __init__(self):
        self.shifts = 0
        self.detects = 0
        self.inserts = 0
        self.removes = 0

    def Add(self, _record):
        self.shifts += _record[0]
        self.detects += _record[1]
        self.inserts += _record[2]
        self.removes += _record[3]
    
    def Show(self):
        print(f'shifts: {self.shifts}, detects: {self.detects}, inserts: {self.inserts}, removes: {self.removes}')
