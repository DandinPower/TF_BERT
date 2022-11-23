from dotenv import load_dotenv
import os
load_dotenv()
VOCAB_SIZE = int(os.getenv('VOCAB_SIZE'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
DROPOUT = float(os.getenv('DROPOUT'))
NUM_HIDDENS = int(os.getenv('NUM_HIDDENS'))
FFN_NUM_INPUT = int(os.getenv('FFN_NUM_INPUT'))
FFN_NUM_HIDDENS = int(os.getenv('FFN_NUM_HIDDENS'))
NORM_SHAPE = [int(os.getenv('NORM_SHAPE'))]
NUM_HEADS = int(os.getenv('NUM_HEADS'))
NUM_LAYERS = int(os.getenv('NUM_LAYERS'))
MAX_LEN = int(os.getenv('MAX_LEN'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

class Config():
    def __init__(self):
        self.vocabSize = VOCAB_SIZE
        self.numHiddens = NUM_HIDDENS
        self.normShape = NORM_SHAPE
        self.ffnNumInput = FFN_NUM_INPUT 
        self.ffnNumHiddens = FFN_NUM_HIDDENS
        self.numHeads = NUM_HEADS
        self.numLayers = NUM_LAYERS
        self.dropout = DROPOUT
        self.maxLen = MAX_LEN
        self.keySize = NUM_HIDDENS
        self.querySize = NUM_HIDDENS
        self.valueSize = NUM_HIDDENS
        self.batchSize = BATCH_SIZE
        self.parameters = None
    
    def SetHyperParameters(self,vocabSize,numHiddens,normShape,ffnNumInput,ffnNumHiddens,numHeads,dropout,maxLen,keySize,querySize,valueSize,batchSize):
        self.vocabSize = vocabSize
        self.numHiddens = numHiddens
        self.normShape = normShape
        self.ffnNumInput = ffnNumInput 
        self.ffnNumHiddens = ffnNumHiddens
        self.numHeads = numHeads
        self.numLayers = numLayers 
        self.dropout = dropout 
        self.maxLen = maxLen
        self.keySize = keySize
        self.querySize = querySize
        self.valueSize = valueSize
        self.batchSize = batchSize
    
    def __str__(self):
        return f'batchSize: {self.batchSize},\nvocabSize: {self.vocabSize},\nnumHiddens: {self.numHiddens},\nnormShape: {self.normShape},\nffnNumInput: {self.ffnNumInput},\nffnNumHiddens: {self.ffnNumHiddens},\nnumHeads: {self.numHeads},\nnumLayers: {self.numLayers},\ndropout: {self.dropout},\nmaxLen: {self.maxLen},\nkeySize: {self.keySize},\nquerySize: {self.querySize},\nvalueSize: {self.valueSize}\n'