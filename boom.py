
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import matplotlib
import numpy as np
import os 
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

def readfile(filename):
    train_data = []
    train_label = []

    input_num = 3

    data_raw = open(filename)
    for line in data_raw:
        line = line.strip()
        train = np.array([np.float32(x)for x in line.split(",")[0:input_num]])
        label = np.int32(line.split(",")[input_num])
        train_data.append(train)
        train_label.append(label)
    return train_data,train_label

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

    return train,label


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

def csvFilter():
    files = []
    for file in find_all_files('./kas/'):
        if(file[-4:] == '.csv'):
            files.append(file)
    return files

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
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

    trainset = []
    labelset = []

    for file in csvFilter():
        t , l= readfile(file)
        trainset.extend(t)
        labelset.extend(l)

    train_data, train_label=rotateDimension(trainset,labelset)

    threshold = np.int32(len(train_data)/2)
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
    #print csvFilter()
    
