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

from sst.seq2seq_model import EncoderRNN, WeightedCrossEntropy, AttnDecoderRNN
#from sst.seq2seq_model import BahdanauAttnDecoderRNN as AttnDecoderRNN

USE_CUDA = True

teacher_forcing_ratio = 0.5
clip = 5.0

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

def parse_args():
    p = argparse.ArgumentParser(
      description="SST example evaluation script",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-mn', '--method_name', default='seq2seq', help='shorthand for method', type=str)

    p.add_argument('-td', '--train-dir',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/train_stride_1_seqlen_128_k_32/',
             help='filepath for train dataset directory', type=str)

    p.add_argument('-vd', '--val-dir',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/val_stride_1_seqlen_128_k_32/',
             help='filepath for val dataset directory', type=str)

    p.add_argument('-pd', '--param_dir', default='/nfs/bigbang/pratik18v/cse599/sst/data/params/',
            help='path to directory containing model parameters', type=str)

    p.add_argument('-ckp', '--checkpoint', default=None, help='checkpoint file to load', type=str)

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

    p.add_argument('-bs', '--batch_size', default=16,
            help='Size of mini batch', type=int)

    p.add_argument('-e', '--num_epochs', default=200,
            help='Maximum iterations for training', type=int)

    p.add_argument('-tt', '--tIoU', default=0.5,
            help='Threshold for tIoU', type=float)

    p.add_argument('-drp', '--dropout', default=0.5,
            help='Dropout probability', type=float)

    p.add_argument('-v', '--verbose', default=False,
            help='filename for output proposals', type=bool)

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

def train(input_batches, target_batches, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    #Put seq_length as first dimension (T X B X D/K)
    input_batches = input_batches.permute(1,0,2)
    target_batches = target_batches.permute(1,0,2)

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_batches.size()[0]
    target_length = target_batches.size()[0]

    assert input_length == target_length

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_batches, encoder_hidden)

    # Prepare input and output variables
    decoder_input = target_batches[0]
    #decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    decoder_hidden = decoder.hidden_linear(encoder_hidden.view(-1, encoder.hidden_size))
    decoder_hidden = decoder_hidden.view(decoder.n_layers, encoder.batch_size, decoder.output_size)

    all_decoder_outputs = Variable(torch.zeros(target_length, encoder.batch_size, decoder.output_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Run through decoder one time step at a time
    for t in range(target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = criterion(
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        torch.clamp(all_decoder_outputs.transpose(0, 1).contiguous(), 0.001, 0.999), # -> batch x seq
        )

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

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
    for fname in val_fnames:
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

#    #Reading val data
#    print 'Reading validation data ...'
#    pbar = tqdm.tqdm(total = len(val_fnames))
#    X_val = []
#    y_val = []
#    for fname in val_fnames:
#        data = sio.loadmat(fname)
#        feat = data['relu6']
#        gt = data['label']
#
#        assert feat.shape[0] == gt.shape[0]
#
#        X_val.append(feat)
#        y_val.append(gt)
#        pbar.update(1)
#    pbar.close()
#
#    X_val = np.asarray(X_val, dtype=np.float64)
#    y_val = np.asarray(y_val)
#
#    y_val[y_val >= args.tIoU] = 1
#    y_val[y_val < args.tIoU] = 0
#    y_val = y_val.astype(np.float64)

    print 'DONE'

    print 'Building model ...'

    # Initialize models
    encoder = EncoderRNN(args.feat_dim, args.width, args.num_proposals, args.depth, args.batch_size)
    decoder = AttnDecoderRNN('general', args.num_proposals, args.num_proposals, args.width, args.depth,
            dropout=args.dropout)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # Initialize optimizers and criterion
    learning_rate = 0.0001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = WeightedCrossEntropy(Variable(torch.from_numpy(w0).cuda()),
        Variable(torch.from_numpy(w1).cuda()))

    print 'DONE'

    # Configuring training
    plot_every = 10
    print_every = 1

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []

    # Begin!
    print 'Started training ... '

    for i in range(args.num_epochs):
        print_loss_total = 0 # Reset every print_every
        plot_loss_total = 0 # Reset every plot_every
        for training_pair in iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True):

            # Get training data for this cycle
            input_batches = Variable(torch.from_numpy(training_pair[0]).cuda())
            target_batches = Variable(torch.from_numpy(training_pair[1]).cuda())

            # Run the train function
            start_time = time.time()
            loss = train(input_batches, target_batches, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion)
            print 'Time per batch: {}'.format(time.time() - start_time)
            print 'Loss: {}'.format(loss)
            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every * train_batches)
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch /
                n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / (plot_every * train_batches)
            plot_losses.append(plot_loss_avg)

if __name__ == '__main__':
    args = parse_args()
    main(args)
