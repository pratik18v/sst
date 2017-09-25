#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 03:34:43 2017

@author: pratik18v
"""
import argparse
import os

import time
import hickle as hkl
import lasagne
import numpy as np

import theano
import theano.tensor as T

from sst.model import SSTSequenceEncoder
from sst.utils import get_segments, nms_detections

def parse_args():
    p = argparse.ArgumentParser(
      description="SST example evaluation script",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-nms', '--nms-thresh', default=0.7, type=float,
                   help='threshold for non-maximum suppression')
    p.add_argument('-td', '--train-dataset', help='filepath for train dataset.',
                   default='data/train/train_rand_data.hkl', type=str)
    p.add_argument('-vd', '--val-dataset', help='filepath for val dataset.',
                   default='data/val/val_rand_data.hkl', type=str)
    p.add_argument('-mp', '--model-params', help='filepath to model params file.',
                   default='data/params/sst_demo_th14_k32.hkl', type=str)
    p.add_argument('-od', '--output-dir', default='data/proposals/output',
                   help='folder filepath for output proposals', type=str)
    p.add_argument('-on', '--output-name', default='results.csv',
                   help='filename for output proposals', type=str)
    p.add_argument('-v', '--verbose', default=False,
                   help='filename for output proposals', type=bool)
    return p.parse_args()

def load_model(filename, input_var=None, target_var=None,  **kwargs):
    model_params = hkl.load(filename)
    arch_params = model_params.get('arch_params', {})
    model = SSTSequenceEncoder(input_var, target_var, **arch_params)
    model.initialize_pretrained(model_params['params'])
    return model

def generate_random_data(num_samples, feat_dim, k):

    time_steps =  [np.random.randint(50,150) for _ in range(num_samples)]
    train_data = []
    train_labels = []
    for i in range(num_samples):
        train_data.append(np.random.rand(time_steps[i], feat_dim).astype(np.float32))
        train_labels.append(np.random.randint(2, size=[time_steps[i], k]))

    return train_data, train_labels

data_index = 0
def generate_batch(X, y, batch_size):
    global data_index
    data = zip(X, y)
    np.random.shuffle(data)
    X, y = zip(*data)
    X, y = list(X), list(y)
    if data_index >= len(X):
        data_index = 0
    batch = tuple([X[data_index:data_index+batch_size], \
                   y[data_index:data_index+batch_size]])
    data_index += batch_size
    return batch

def main(args):
    batch_size = 32
    max_steps = 60000
    num_samples = 1000
    num_vals = 50
    feat_dim = 500
    k = 32

    if not os.path.isfile(args.train_dataset):
        print 'Generating random train data ...'
        train_data, train_labels = \
            generate_random_data(num_samples, feat_dim, k)
        with open(args.train_dataset, 'w') as f:
            hkl.dump([train_data, train_labels], f)
    else:
        print 'Loading train dataset ...'
        with open(args.train_dataset, 'r') as f:
            train_data, train_labels = hkl.load(f)

    if not os.path.isfile(args.val_dataset):
        print 'Generating random val data ...'
        val_data, val_labels = \
            generate_random_data(num_vals, feat_dim, k)
        with open(args.val_dataset, 'w') as f:
            hkl.dump([val_data, val_labels], f)
    else:
        print 'Loading val dataset ... '
        with open(args.val_dataset, 'r') as f:
            val_data, val_labels = hkl.load(f)

    print 'DONE'
    print 'Building model ...'
    input_var = T.tensor3('inputs')
    target_var = T.lmatrix('target')
    sst_model = load_model(args.model_params, input_var=input_var, target_var=target_var)
    sst_model.compile()
    print 'DONE'

    print 'Starting training ...'
    global_start = time.time()
    for i in range(max_steps):
        X_batch, y_batch = generate_batch(train_data, train_labels, batch_size)
        train_err = 0
        start_time = time.time()
        for j in range(batch_size):
            X_t = np.expand_dims(X_batch[j], axis=0)
            y_t = y_batch[j]
            # obtain proposals
            train_err += sst_model.forward_eval(X_t, y_t, mode='train')

        val_err = 0
        for j in range(num_vals):
            X_t = np.expand_dims(val_data[j], axis=0)
            y_t = val_labels[j]
            # obtain proposals
            err = sst_model.forward_eval(X_t, y_t, mode='test')
            val_err += err

        print("Global iteration {} of {} took {:.3f}s".format(
            i + 1, max_steps, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / batch_size))
        print("  validation loss:\t\t{:.6f}".format(val_err / num_vals))

    print 'DONE'
    print 'Total training time: {}'.format(time.time() - global_start)

if __name__ == '__main__':
    args = parse_args()
    main(args)

