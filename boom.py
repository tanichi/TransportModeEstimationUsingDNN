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

def findAllFiles(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

def csvFilter(directory):
    files = []
    print "using files"
    for file in findAllFiles('./'+directory):
        if(file[-4:] == '.csv'):
            print file
            files.append(file)
    return files

def readfile(filenames):
    train_data = []
    train_label = []

    input_num = 3
    for file in filenames:
        data_raw = open(file)
        for line in data_raw:
            line = line.strip()
            train = np.array([np.float32(x)for x in line.split(",")[0:input_num]])
            label = np.int32(line.split(",")[input_num])

            train_data.append(train)
            train_label.append(label)
    return np.array(train_data),np.array(train_label)

def rotateDimension(train,label):
    originalTrain = np.array(train)
    originalLabel = label
    train = []
    label = []

    train.extend(originalTrain[:,[0,1,2]])
    train.extend(originalTrain[:,[0,2,1]])    
    train.extend(originalTrain[:,[1,0,2]])
    train.extend(originalTrain[:,[1,2,0]])
    train.extend(originalTrain[:,[2,0,1]])
    train.extend(originalTrain[:,[2,1,0]])

    label.extend(originalLabel) 
    label.extend(originalLabel)
    label.extend(originalLabel)
    label.extend(originalLabel)
    label.extend(originalLabel)
    label.extend(originalLabel)

    return np.array(train),np.array(label)

def shuffleTrainData(trainset, labelset):
    labelset = np.array(labelset).reshape(1,len(labelset))
    trainset = np.array(trainset)
    #print labelset.shape
    merge =  np.concatenate((trainset, labelset.T), axis=1) # merge

    np.random.shuffle(merge)
    #print merge

    return merge[:,[0,1,2]].astype(np.float32) , merge[:,3].astype(np.int32)

def dataHistogram(label):
    label = np.array(label)
    his = np.histogram(label,bins=int(label.max())+1,range=(-0.5,label.max()+0.5))
    for i in range(len(his[0])):
        print ("class {0} data is {1}".format(i,his[0][i]))

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--trainfile', '-t', type=str, default='test', help='training data file path')
    parser.add_argument('--batchsize', '-b', type=int, default=500, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=20, help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false', help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    
    model = L.Classifier(MLP(args.unit, 10))
    '''if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    '''

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_data, train_label = readfile(csvFilter(args.trainfile))
    train_data, train_label = rotateDimension(train_data,train_label)
    threshold = np.int32(len(train_data)/2)

    
    print train_data.dtype
    print train_label.dtype

    train_data, train_label = shuffleTrainData(train_data, train_label)

    print train_data.dtype
    print train_label.dtype

    #sys.exit()
    train = chainer.datasets.TupleDataset(train_data[0:threshold], train_label[0:threshold])
    test  = chainer.datasets.TupleDataset(train_data[threshold:],  train_label[threshold:])

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,repeat=False, shuffle=False)
    
    # Set up a trainer
    updater = training.updaters.StandardUpdater(train_iter,optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
   main()
    
    

