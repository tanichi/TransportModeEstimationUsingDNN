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
import trainingdata

matplotlib.use('Agg')

# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_units)  # n_units -> n_units
            self.l4 = L.Linear(None, n_units)  # n_units -> n_units
            self.l5 = L.Linear(None, n_units)  # n_units -> n_units
            self.l6 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        return self.l6(h5)

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--trainfile', '-t', type=str, default='test', help='training data file path')
    parser.add_argument('--validationfile', '-v', type=str, default='test', help='training validation data file path')
    parser.add_argument('--batchsize', '-b', type=int, default=1000, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=20, help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false', help='Disable PlotReport extension')
    parser.add_argument('--snapshot', '-s', type=str, default='result/snapshot_iter_281', help='use model snapshot')
    args = parser.parse_args()


    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    
    model = L.Classifier(MLP(args.unit, 10))
    '''if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    '''

    chainer.serializers.load_npz(args.snapshot, model,path='updater/model:main/')
    with chainer.using_config('train', False):
        x = np.asarray([0.0,-9.1,-0.7],dtype=np.float32).reshape(1,3)
        #print(np.shape(x))
        y = model.predictor(x).array
    print('予想ラベル:{0}'.format(y.argmax(axis=1)[0]))
    
    
if __name__ == '__main__':
    main()

