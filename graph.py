import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np     
from dotenv import load_dotenv
import os
load_dotenv()

IMAGE_PATH = os.getenv('IMAGE_PATH')

def AddList(A, B):
    C = []
    for a, b in zip(A,B):
        C.append(a + b)
    return C

def GetDistributionFromFile(_filename, _type):
    distribution = []
    with open(_filename, "r") as txt_file:
        readline=txt_file.read().splitlines()
        for i in range(len(readline)):
            distribution.append(_type(readline[i]))
    return distribution

def PlotDistribution(_distributions,_labels, _xlabel, _ylabel, _title):
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 20)
    for distribution, label in zip(_distributions, _labels):
        plt.plot(distribution, label = label)
    plt.xlabel(_xlabel)
    plt.ylabel(_ylabel)
    plt.title(_title)
    plt.legend()
    plt.savefig(f'{IMAGE_PATH}{_title}.png')
    plt.close()

def PlotComparison(A, _labelA, B, _labelB, _title, _yscale):
    col_count = 4                  # 由於有3個月，設定類別基數為3
    bar_width = 0.1                # 設定長條圖每個長條寬度
    index = np.arange(col_count)   # 依據3個類別(3個月)設定索引值，便於後續長條圖的位置設定
    A = plt.bar(index,         # 索引值代表A肇事原因的長條位置，如index=[0,1,2]，分別在3個月的第一個位置
           A,              # 設定長條圖的資料 
           bar_width,      # 設定長條寬度
           alpha=1,       # 設定透明度
           label=_labelA)# 設定標籤 
    B = plt.bar(index+0.1,     # 索引值為A的索引值+0.2(長條寬度)，顯示於A長條的旁邊一個長條寬度的位置
            B,
            bar_width,
            alpha=1,
            label=_labelB)
    plt.ylabel("Numbers Of SKRM Operation")
    plt.yscale(_yscale)    
    plt.xlabel("Types Of SKRM Operation")       
    plt.title(_title)  # 設定標題、文字大小
    plt.xticks(index+ .05 ,("shifts", "detects", "removes", "inserts"))  #.xticks為x軸文字(為了置中所以+0.3/2)
    #plt.ylim(0, 250)                 # 設定y軸範圍
    plt.legend(prop = {'size':9})    # 設定圖例及其大小
    #plt.grid(True)                   # 顯示格線
    #plt.show()
    # 儲存圖檔
    #plt.savefig("Bar chart of car accident.jpg",   # 儲存圖檔
    #            bbox_inches='tight',               # 去除座標軸占用的空間
    #            pad_inches=0.0)                    # 去除所有白邊
    #plt.close()      # 關閉圖表
    #plt.show()
    plt.savefig(f'{IMAGE_PATH}{_title}.png')
    plt.close()

def qqp():
    original = GetDistributionFromFile('train/history/qqp_original_32_20.txt',float)
    distributions = [original]
    labels = ['Original(0)']
    PlotDistribution(distributions, labels, 'epochs', 'accuracy', 'qqp')

def cola():
    original = GetDistributionFromFile('train/history/cola_original_32_20.txt',float)
    distributions = [original]
    labels = ['Original(0)']
    PlotDistribution(distributions, labels, 'epochs', 'accuracy', 'cola')

def sst():
    original = GetDistributionFromFile('train/history/sst_original_32_20.txt',float)
    distributions = [original]
    labels = ['Original(0)']
    PlotDistribution(distributions, labels, 'epochs', 'accuracy', 'sst')

def stsb():
    original = GetDistributionFromFile('train/history/stsb_original_32_20.txt',float)
    distributions = [original]
    labels = ['Original(0)']
    PlotDistribution(distributions, labels, 'epochs', 'accuracy', 'stsb')

if __name__ == '__main__':
    qqp()
    cola()
    sst()
    stsb()

