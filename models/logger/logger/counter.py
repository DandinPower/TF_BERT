import numpy as np 
import tensorflow as tf 

class Counter:
    def __init__(self, *args):
        self.id = args[0]
        self.tensors = []
        self.floatNumsInTensors = []
        self.N = 32

    def SetLog(self, _tensors):
        self.tensors.clear()
        for tensor in _tensors:
            temp = []
            for dim in tensor:
                temp.append(dim)
            self.tensors.append(temp)
        self.Init()

    def ShowLog(self):
        pass

    def Init(self):
        pass

    #當有新資料設定進來時就要重算一遍
    def CountFloatNums(self):
        self.floatNumsInTensors.clear()
        for tensor in self.tensors:
            temp = 1
            for dim in tensor:
                temp *= dim 
            self.floatNumsInTensors.append(temp)
            
    def GetSkrmNaiveRecord(self):
        pass

    def GetSkrmImproveRecord(self):
        pass

class MatmulCounter(Counter):
    def __init__(self, *args):
        self.standardTensor = []
        super().__init__(*args)

    def SetBlockSize(self, _blockSize):
        self.blockSize = _blockSize
        #blocksize[[m, n], [n, y]]

    def Init(self):
        self.StandardTensorDimension()
        self.CountFloatNums()
        self.CountJ()
        self.CountI()
        self.CountK()

    #將3維轉二維
    def StandardTensorDimension(self):
        self.standardTensor.clear()
        for tensor in self.tensors:
            if len(tensor) <= 2:
                self.standardTensor.append(tensor)
            else:
                firstDims = 1
                for i in range(len(tensor)-1):
                    firstDims *= tensor[i]
                self.standardTensor.append([firstDims, tensor[-1]])
    
    #計算大I 
    def CountI(self):
        firstDim = self.standardTensor[0][0]
        firstBlockDim = self.blockSize[0][0]
        self.I =  firstDim // firstBlockDim
        if firstDim % firstBlockDim != 0:
            self.I += 1

    #計算大J
    def CountJ(self):
        lastDim = self.tensors[-1][-1]
        lastBlockDim = self.blockSize[-1][-1]
        self.J =  lastDim // lastBlockDim
        if lastDim % lastBlockDim != 0:
            self.J += 1

    #計算大K
    def CountK(self):
        commonDim = self.standardTensor[0][-1]
        commonBlockDim = self.blockSize[0][-1]
        self.K = commonDim // commonBlockDim
        if commonDim % commonBlockDim != 0:
            self.K += 1

    def ShowIJK(self):
        print(f'matmul: {self.tensors[0]},{self.tensors[1]}; standard: {self.standardTensor[0]},{self.standardTensor[1]}; IKJ: {self.I}, {self.K}, {self.J}')

    def ShowLog(self):
        return f'matmul;{self.id};{self.tensors[0]};{self.tensors[1]}'

    def GetSkrmNaiveRecord(self):
        shifts = 0
        detects = 0
        shifts += 2 * self.J * self.N * self.floatNumsInTensors[0]
        shifts += 2 * self.I * self.N * self.floatNumsInTensors[1]
        detects += self.J * self.N * self.floatNumsInTensors[0]
        detects += self.I * self.N * self.floatNumsInTensors[1]
        return [shifts, detects, 0, 0]

    def GetSkrmImproveRecord(self):
        shifts = 0
        detects = 0
        if self.J % 4 == 0:
            shifts += self.J * self.N * self.floatNumsInTensors[0]
        else:
            shifts = (self.J + (4 - (self.J % 4))) * self.N * self.floatNumsInTensors[0]
        shifts += self.I * self.N * self.floatNumsInTensors[1]
        detects += self.J * self.N * self.floatNumsInTensors[0]
        detects += self.I * self.N * self.floatNumsInTensors[1]
        return [shifts, detects, 0, 0]

class TransposeCounter(Counter):
    def __init__(self, *args):
        super().__init__(*args)

    def ShowLog(self):
        return f'transpose;{self.id};{self.tensors[0]}'

    def Init(self):
        self.CountFloatNums()

    def GetSkrmNaiveRecord(self):
        #需要刪除/新增所有的floats bits 
        shifts = 2 * self.N * self.floatNumsInTensors[0]
        #不需要知道原來的值是什麼 
        detects = 0
        #由於Approximate的機制所以只需寫入一半的Skyrmions
        inserts = int(self.N / 2) * self.floatNumsInTensors[0]
        #把原本的Skyrmions刪掉
        removes = int(self.N / 2) * self.floatNumsInTensors[0]
        return [shifts, detects, inserts, removes]

    def GetSkrmImproveRecord(self):
        return [0, 0, 0, 0]