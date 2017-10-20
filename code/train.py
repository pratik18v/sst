import argparse
import os

import tqdm
import time
import pickle as pkl
import hickle as hkl
import numpy as np
import scipy.io as sio

import torch
import torch.optim as optim
from torch.autograd import Variable

from sst.model import SSTSequenceEncoder, StatePairwiseConcat, WeightedCrossEntropy

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

def parse_args():
    p = argparse.ArgumentParser(
      description="SST Training script",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-mn', '--method_name', default='baseline', help='shorthand for method',type=str)

    #Directoires
    p.add_argument('-td', '--train-dir',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/train_stride_1_seqlen_128_k_32/',
            help='filepath for train dataset directory', type=str)

    p.add_argument('-vd', '--val-dir',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/val_stride_1_seqlen_128_k_32/',
            help='filepath for val dataset directory', type=str)

    p.add_argument('-pd', '--param_dir', default='/nfs/bigbang/pratik18v/cse599/sst/data/params/',
            help='path to directory containing model parameters', type=str)

    p.add_argument('-ckp', '--checkpoint', default='epoch_10.pth.tar', help='checkpoint file to load', type=str)

    #Model parameters
    p.add_argument('-k', '--num_proposals', default=32, help='Number of proposals generated at each timestep',
            type=int)

    p.add_argument('-sl', '--seq_length', default=128,  help='Sequence length of each training instance',
            type=int)

    p.add_argument('-dp', '--depth', default=1, help='Number of recurrent layers in sequence encoder', type=int)

    p.add_argument('-w', '--width', default=256, help='Size of hidden state in each recurrent layer', type=int)

    p.add_argument('-fd', '--feat-dim', default=500, help='Dimension of c3d features', type=int)

    #Training parameters
    p.add_argument('-bs', '--batch_size', default=256, help='Size of mini batch', type=int)

    p.add_argument('-lr', '--learning_rate', default=0.001, help='Learning rate for optimization', type=float)

    p.add_argument('-mom', '--momentum', default=0.9, help='Momentum for optimization', type=float)

    p.add_argument('-cl', '--clip', default=100, help='Gradient clipping parameter', type=float)

    p.add_argument('-e', '--num_epochs', default=200, help='Maximum iterations for training', type=int)

    p.add_argument('-tt', '--tIoU', default=0.5, help='Threshold for tIoU', type=float)

    p.add_argument('-drp', '--dropout', default=0.5, help='Dropout probability', type=float)

    p.add_argument('-v', '--verbose', default=False, help='filename for output proposals', type=bool)

    return p.parse_args()

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(args):

    #Writing configuration file
    ckpt_dir = args.param_dir + args.method_name + '/'
    if os.path.exists(ckpt_dir) == False:
        os.mkdir(ckpt_dir)
    f = open(ckpt_dir + 'configs.txt', 'w')
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))
    f.close()

    #Listing data files
    train_fnames = []
    for fname in os.listdir(args.train_dir):
        train_fnames.append(args.train_dir + fname)

    val_fnames = []
    for fname in os.listdir(args.val_dir):
        val_fnames.append(args.val_dir + fname)

    print 'Number of training samples: {}'.format(len(train_fnames))
    print 'Number of validation samples: {}'.format(len(val_fnames))

    #Reading train data
    print 'Reading training data ...'
    X_train = []
    y_train = []
    pbar = tqdm.tqdm(total = len(train_fnames))
    for fname in train_fnames:
        data = sio.loadmat(fname)
        feat = data['relu6']
        gt = data['label']

        assert feat.shape[0] == gt.shape[0]

        X_train.append(feat)
        y_train.append(gt)
        pbar.update(1)
    pbar.close()

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    y_train[y_train >= args.tIoU] = 1
    y_train[y_train < args.tIoU] = 0
    y_train = y_train.astype(np.float64)

    w0 = np.mean(((y_train == 1).sum(1).astype(np.float64) / args.seq_length), axis=0)
    w1 = np.mean(((y_train == 0).sum(1).astype(np.float64) / args.seq_length), axis=0)

    print 'DONE ... '

    #Reading val data
    print 'Reading validation data ...'
    pbar = tqdm.tqdm(total = len(val_fnames))
    X_val = []
    y_val = []
    for fname in val_fnames:
        data = sio.loadmat(fname)
        feat = data['relu6']
        gt = data['label']

        assert feat.shape[0] == gt.shape[0]

        X_val.append(feat)
        y_val.append(gt)
        pbar.update(1)
    pbar.close()

    X_val = np.asarray(X_val, dtype=np.float64)
    y_val = np.asarray(y_val)

    y_val[y_val >= args.tIoU] = 1
    y_val[y_val < args.tIoU] = 0
    y_val = y_val.astype(np.float64)

    print 'DONE'

    print 'Building model ...'

    model = SSTSequenceEncoder(feature_dim = args.feat_dim, hidden_dim = args.width,
            seq_length = args.seq_length, batch_size = args.batch_size, num_proposals = args.num_proposals,
            num_layers = args.depth, dropout = args.dropout)
#    model = StatePairwiseConcat(feature_dim = args.feat_dim, hidden_dim = args.width,
#            seq_length = args.seq_length, batch_size = args.batch_size, num_proposals = args.num_proposals,
#            num_layers = args.depth, dropout = args.dropout)
    model.cuda()
    criterion = WeightedCrossEntropy(Variable(torch.from_numpy(w0).cuda()),
                    Variable(torch.from_numpy(w1).cuda()))
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum)

   # optionally resume from a checkpoint
    if args.checkpoint:
        if os.path.isfile(ckpt_dir + args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(ckpt_dir + args.checkpoint)
            args.start_epoch = checkpoint['epoch']
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.checkpoint, checkpoint['epoch']))
            print("Train loss at this epoch: {}".format(train_loss))
            print("Val loss at this epoch: {}".format(val_loss))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_dir + args.checkpoint))

    print 'DONE'

    print 'Starting training ...'

    global_start = time.time()
    f_loss = open(ckpt_dir + 'loss.txt', 'w')
    for i in range(args.num_epochs):
        start_time = time.time()
        train_loss = 0.0
        train_batches = 0
        model.train(True)
        #pbar = tqdm.tqdm(total = int(len(X_train) / args.batch_size))
        for batch in iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True):
            model.zero_grad()
            model.hidden = model.init_hidden()

            X_t, y_t = batch
            X_t, y_t = Variable(torch.from_numpy(X_t).cuda()), Variable(torch.from_numpy(y_t).cuda())

            outputs, states = model(X_t)
            loss = criterion(y_t, outputs)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            train_loss += loss.data[0]
            train_batches += 1
            #pbar.update(1)
        #pbar.close()

        train_loss_str = 'train loss: \t\t %.5f' %(train_loss / train_batches)
        print train_loss_str
        f_loss.write(train_loss_str)

        if i%5 == 0:
            val_loss = 0.0
            val_batches = 0
            model.train(False)
            #pbar = tqdm.tqdm(total = int(len(X_val) / args.batch_size))
            for batch in iterate_minibatches(X_val, y_val, args.batch_size, shuffle=False):

                model.hidden = model.init_hidden()

                X_t, y_t = batch
                X_t, y_t = Variable(torch.from_numpy(X_t).cuda()), Variable(torch.from_numpy(y_t).cuda())

                outputs, states = model(X_t)
                loss = criterion(y_t, outputs)
                val_loss += loss.data[0]
                val_batches += 1
                #pbar.update(1)
            #pbar.close()

            val_loss_str = 'val loss: \t\t %.5f' %(val_loss / val_batches)
            print val_loss_str
            f_loss.write(val_loss_str)

        print('[{},{}] Time took: {:.4f}'.format(i+1, args.num_epochs, time.time() - start_time))

        #Save model after every 10 epochs
        if i % 10 == 0:
            print 'Saving parameters ... '
            torch.save({
                'epoch': i,
                'arch': args.method_name,
                'state_dict': model.state_dict(),
                'train_loss': train_loss / train_batches,
                'val_loss': val_loss / val_batches,
                'optimizer' : optimizer.state_dict(),
            }, ckpt_dir + 'epoch_' + str(i) + '.pth.tar')
            print 'Saved at: {}'.format(ckpt_dir + 'epoch_' + str(i) + '.pth.tar')


    print 'DONE'
    f_loss.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
