# -*- coding: utf-8 -*-

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import matplotlib
import numpy as np
import os 
import sys
import itertools

matplotlib.use('Agg')

class TrainData():
    def __init__(self,directory):
        self.data = np.empty(0)
        self.label = np.empty(0)

        csvFiles = []
        print "using files"
        for root, dirs, files in os.walk(directory):
            for file in files:
                if(file[-4:] == '.csv'):
                    print os.path.join(root, file)
                    csvFiles.append(os.path.join(root, file))

        train = []
        label = []                    
        input_num = 3
        for file in csvFiles:
            data_raw = open(file)
            for line in data_raw:
                line = line.strip()
                train.append(np.array([np.float32(x)for x in line.split(",")[0:input_num]]))
                label.append(np.int32(line.split(",")[input_num]))
        self.data = np.array(train)
        self.label = np.array(label)

    def rotateDimension(self):
        print("RotateDimension is enable")
        originalTrain = self.data
        originalLabel = self.label
        train = []
        label = []

        for i in list(itertools.permutations(tuple(range(3)))):
            train.extend(originalTrain[:,i])
            label.extend(originalLabel)

        self.data = np.array(train)
        self.label = np.array(label)

    def shuffleTrainData(self):
        print("shuffle train data is enable")
        merge =  np.concatenate((self.data, np.reshape(self.label,(self.label.shape[0],1))), axis=1) # merge
        np.random.shuffle(merge)

        self.data = merge[:,[0,1,2]].astype(np.float32)
        self.label = merge[:,3].astype(np.int32)

    def dataHistogram(self):
        his = np.histogram(self.label,bins=int(self.label.max())+1,range=(-0.5,self.label.max()+0.5))
        for i in range(len(his[0])):
            print ("class {0} data is {1}".format(i,his[0][i]))

    def makeDataset(self):
        return chainer.datasets.TupleDataset(self.data, self.label)
