import torch 
import collections
import os 
from torch import nn
from dotenv import load_dotenv
import pickle
load_dotenv()
DATA_DIR = os.getenv('PRETRAIN_DIR_PATH')
PARAMETER_PATH = os.getenv('PARAMETER_PATH')

def save_variable(v,filename):
  f=open(filename,'wb')
  pickle.dump(v,f)
  f.close()
  return filename

def LoadPretrainParameters():
    parameters = torch.load(os.path.join(DATA_DIR,'pretrained.params'))
    parameters_numpy = {}
    for key,value in parameters.items():
        print(key,value.shape)
        parameters_numpy[key] = value.numpy()
    save_variable(parameters_numpy, PARAMETER_PATH)

if __name__ == '__main__':
    LoadPretrainParameters()