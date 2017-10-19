#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 03:34:43 2017

@author: pratik18v
"""
import argparse
import os

import tqdm
import time
import pickle as pkl
import hickle as hkl
import lasagne
import numpy as np
import scipy.io as sio
from collections import Counter

import theano
import theano.tensor as T

#import matplotlib.pyplot as plt

from sst.model import SSTSequenceEncoder
from sst.utils import get_segments, nms_detections

def parse_args():
    p = argparse.ArgumentParser(
      description="SST example evaluation script",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-mn', '--method_name', default='weighted_log_loss', help='shorthand for method', type=str)

    p.add_argument('-td', '--train-dir',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/train_stride_1_seqlen_128_k_32/',
             help='filepath for train dataset directory', type=str)

    p.add_argument('-vd', '--val-dir',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/val_stride_1_seqlen_128_k_32/',
             help='filepath for val dataset directory', type=str)

    p.add_argument('-pd', '--param_dir', default='/nfs/bigbang/pratik18v/cse599/sst/data/params/',
            help='path to directory containing model parameters', type=str)

    p.add_argument('-k', '--num_proposals', default=32,
            help='Number of proposals generated at each timestep', type=int)

    p.add_argument('-sl', '--seq_length', default=128,
            help='Sequence length of each training instance', type=int)

    p.add_argument('-dp', '--depth', default=1,
            help='Number of recurrent layers in sequence encoder', type=int)

    p.add_argument('-w', '--width', default=256,
            help='Size of hidden state in each recurrent layer', type=int)

    p.add_argument('-fd', '--feat-dim', default=500,
            help='Dimension of c3d features', type=int)

    p.add_argument('-bs', '--batch_size', default=128,
            help='Size of mini batch', type=int)

    p.add_argument('-e', '--num_epochs', default=500,
            help='Maximum iterations for training', type=int)

    p.add_argument('-tt', '--tIoU', default=0.5,
            help='Threshold for tIoU', type=float)

    p.add_argument('-drp', '--dropout', default=0.5,
            help='Dropout probability', type=float)

    p.add_argument('-v', '--verbose', default=False,
            help='filename for output proposals', type=bool)

    return p.parse_args()

def load_model(input_var=None, target_var=None, args=None, w0=None, w1=None,  **kwargs):
    model = SSTSequenceEncoder(input_var, target_var, seq_length=args.seq_length, depth=args.depth,
        width=args.width, num_proposals=args.num_proposals, input_size=args.feat_dim, dropout=args.dropout,
        w0=w0, w1=w1)
    return model

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

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    yb_train = y_train
    yb_train[yb_train >= args.tIoU] = 1
    yb_train[yb_train < args.tIoU] = 0
    yb_train = yb_train.astype(int)

    w0 = np.mean(((yb_train == 1).sum(1).astype(float) / args.seq_length), axis=0)
    w1 = np.mean(((yb_train == 0).sum(1).astype(float) / args.seq_length), axis=0)

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

    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    #y_val[y_val >= args.tIoU] = 1
    #y_val[y_val < args.tIoU] = 0
    #y_val = y_val.astype(int)

    print 'DONE'

    print 'Building model ...'

    input_var = T.tensor3('inputs')
    target_var = T.dtensor3('target')
    sst_model = load_model(input_var=input_var, target_var=target_var, args=args, w0=w0, w1=w1)
    sst_model.compile()

    print 'Loading parameters (if exist) ...'
    filename = args.param_dir + 'params_sl{}_np{}_d{}_w{}_fd{}_{}.hkl'.format(args.seq_length, args.num_proposals, args.depth, args.width, args.feat_dim, args.method_name)
    if os.path.exists(filename):
        sst_model.load_model_params(filename)
    else:
        print 'No parameters found!'
    print 'DONE'

    print 'Starting training ...'

    global_start = time.time()
    f_loss = open(args.method_name + '_loss.txt', 'w')
    for i in range(args.num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True):
            X_t, y_t = batch
            err = sst_model.forward_eval(X_t, y_t)
            train_err += err
            train_batches += 1

        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, args.batch_size, shuffle=False):
            X_t, y_t = batch
            err = sst_model.forward_eval(X_t, y_t)
            val_err += err
            val_batches += 1

        loss_str = '{}, {}\n'.format(train_err / train_batches, val_err / val_batches)
        f_loss.write(loss_str)
        print("Epoch {} of {} took {:.3f}s".format(
                i + 1, args.num_epochs, time.time() - start_time))
        print("training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("validation loss:\t\t{:.6f}".format(val_err / val_batches))

    f_loss.close()
    print 'DONE'
    print 'Total training time: {}'.format(time.time() - global_start)

    print 'Saving configuration and model parameters ...'

    f = open(args.param_dir + '{}_configs'.format(args.method_name), 'w')
    for arg in vars(args):
        f.write('{}: {}\n'.format(arg, getattr(args, arg)))
    f.close()
    sst_model.save_model_params(filename)

    print 'DONE'

if __name__ == '__main__':
    args = parse_args()
    main(args)
