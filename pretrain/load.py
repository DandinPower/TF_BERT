import pickle
import tensorflow as tf
import os 

def load_variable(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

def LoadModel(savePath):
    newModel = tf.saved_model.load(savePath)
    return newModel 

def SaveModel(model,savePath):
    tf.saved_model.save(model,savePath)

class Parameters():
    def __init__(self,data):
        self.data = data

    def __getitem__(self,key):
        return self.data[key]
