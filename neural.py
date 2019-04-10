
import pylab
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from math import exp

def toInt(x):
    return int.from_bytes(x, byteorder = 'big', signed=False)

def loadTask():
    tasks = []
    ans = []
    with open("train-images.idx3-ubyte", 'rb') as f:
        f.read(4)
        n = toInt(f.read(4))
        w = toInt(f.read(4))
        h = toInt(f.read(4))

        tasks = np.zeros((n, w*h))

        for i in range(n):
            for x in range(w):
                for y in range(h):
                    tasks[i][x*w + y] = toInt(f.read(1))

    with open("train-labels.idx1-ubyte", 'rb') as f:
        f.read(4)
        n = toInt(f.read(4))
        ans = np.zeros(n)
        for i in range(n):
            ans[i] = toInt(f.read(1))
    return tasks, ans

def main():
    tasks, ans = loadTask()
    l0 = np.zeros((2,784))
    l1 = np.zeros((2,10))
    l2 = np.zeros((2,4))

    w01 = np.zeros((10, 784))
    w12 = np.zeros((4,10))
    for i in range(len(tasks)):
        l0[0] = tasks[i]
        nextLayer(l0, l1, w01)
        nextLayer(l1, l2, w12)
        
        good = getGoodAns(ans[i])
        for i in range(len(l2[1])):
            l2[1][i] = good[i] - l2[0][i]
        findError(l1, l2, w12)
        setError(l0, l1, w01)
        setError(l1, l2, w12)

def nextLayer(lx, ly, wxy):
    for y in range(len(ly[0])):
        for x in range(len(lx[0])):
            ly[0][y] += lx[0][x] * wxy[y][x]
        ly[0][y] = 1/(1 + exp(-ly[0][y]))

def findError(lx, ly, wxy):
    for x in range(len(lx[1])):
        for y in range(len(ly[1])):
            lx[1][x] += ly[1][y] * wxy[y][x]

def getGoodAns(x):
    kek = x
    res = np.zeros(4)
    for i in range(4):
        res[i] = (kek%2)
        kek = kek // 2
    return res

def setError(lx, ly, wxy):
    for y in range(len(ly[1])):
        f = ly[0][y] * (1-ly[0][y])
        for x in range(len(lx[1])):
            wxy[y][x] += lx[1][x]*f*lx[0][x] 
main()