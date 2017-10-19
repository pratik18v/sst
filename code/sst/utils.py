#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 03:34:43 2017

@author: pratik18v
"""
import os
import argparse
import numpy as np
import pickle as pkl
import hickle as hkl
import scipy.io as sio
from sklearn.decomposition import PCA

def parse_args():
    p = argparse.ArgumentParser(
      description="Utility code to prepare dataset for training.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-fd', '--feats-dir', help='C3D video features directory',
            default='/nfs/bigeye/yangwang/DataSets/THUMOS14/src/extract_c3d/c3d_feat/', type=str)

    p.add_argument('-pfd', '--pca-feats-dir', help='C3D video features with PCA directory',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/c3d_pca/', type=str)

    p.add_argument('-sd', '--stats-data', help='Path to save stats for PCA (U and mean)',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/pca_c3d_relu6_thumos.pkl', type=str)

    p.add_argument('-dd', '--data-dir', help='Directory storing all the data',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/', type=str)

    p.add_argument('-m', '--mode', help='mode for which PCA is taking place [validation, test]',
            default='test', type=str)

    p.add_argument('-s', '--stride', help='Stride for densely sampling data', default=1, type=int)

    p.add_argument('-sl', '--seq_len', help='Length of sequence of each data point', default=128, type=int)

    p.add_argument('-fdim', '--feat_dim', help='Dimension of c3d feature dimension', default=500, type=int)

    p.add_argument('-k', '--num_proposals', help='Number of proposals generated at each timestep',
            default=32, type=int)

    return p.parse_args()

def perform_pca(args):

    fnames = []
    for fname in os.listdir(args.feats_dir):
        if fname.split('_')[1] == args.mode:
            fnames.append(args.feats_dir + fname)
        elif fname.split('_')[1] == args.mode:
            fnames.append(args.feats_dir + fname)

    if os.path.exists(args.stats_data) == False:
        #Computing PCA statistics
        feats_concat = sio.loadmat(fnames[0])['relu6']

        ctr = 0
        for fname in fnames[1:]:
            print ctr
            ctr += 1
            feats = sio.loadmat(fname)['relu6']
            feats_concat = np.concatenate((feats_concat, feats), axis=0)

        mean = np.mean(feats_concat, axis=0)
        #Whitening
        feats_concat = np.subtract(feats_concat, mean)

        #PCA
        pca = PCA(n_components=args.feat_dim, whiten=True)
        pca.fit(feats_concat)
        with open(args.stats_data,'w') as f:
            pkl.dump([pca, mean], f)

    else:
        with open(args.stats_data, 'r') as f:
            pca, mean = pkl.load(f)
        print mean.shape

    #Applying transformation
    for fname in fnames:
        feats = sio.loadmat(fname)['relu6']
        feats = np.subtract(feats, mean)
        feats_proj = pca.transform(feats)

        if os.path.exists(args.data_dir + 'c3d_pca') == False:
            os.mkdir(args.data_dir + 'c3d_pca')
        sio.savemat(args.data_dir + 'c3d_pca/' + fname.split('/')[-1], {'relu6': feats_proj})

def sample_data(args):

    train_fnames = []
    for fname in os.listdir(args.pca_feats_dir):
        if fname.split('_')[1] == 'validation':
            train_fnames.append(args.pca_feats_dir + fname)

    #print len(set([tf.split('/')[-1].split('.')[0] for tf in train_fnames]))
    gt_data = sio.loadmat(args.data_dir + 'val_ground_truths_k32.mat')['ground_truth'][0,0]

    #Making directories to store train and val data
    train_dir = 'train_stride_{}_seqlen_{}_k_{}'.format(args.stride, args.seq_len, args.num_proposals)
    if os.path.exists(args.data_dir + train_dir) == False:
        os.mkdir(args.data_dir + train_dir)
    val_dir = 'val_stride_{}_seqlen_{}_k_{}'.format(args.stride, args.seq_len, args.num_proposals)
    if os.path.exists(args.data_dir + val_dir) == False:
        os.mkdir(args.data_dir + val_dir)

    for fname in train_fnames:
        if train_fnames.index(fname) <= 0.8 * len(train_fnames):
            direc = train_dir
            #print len(set([tf.split('/')[-1].split('_')[2] for tf in os.listdir(args.data_dir + direc)]))
        else:
            direc = val_dir
            #print len(set([tf.split('/')[-1].split('_')[2] for tf in os.listdir(args.data_dir + direc)]))

        vid_id = fname.split('/')[-1].split('.')[0]
        feat = sio.loadmat(fname)['relu6']
        gt = gt_data[vid_id]

        if feat.shape[0] < gt.shape[0]:
            gt = gt[:feat.shape[0],:]
        elif feat.shape[0] > gt.shape[0]:
            diff = feat.shape[0] - gt.shape[0]
            endcol = gt[-1,:]
            padding = np.tile(endcol, (diff,1))
            gt = np.vstack((gt, padding))

        assert feat.shape[0] == gt.shape[0]

        #If time steps less than window length, replicate last time step
        if feat.shape[0] < args.seq_len:
            feat = np.vstack((feat, np.tile(feat[-1,:], (args.seq_len - feat.shape[0], 1))))
            gt = np.vstack((gt, np.tile(gt[-1,:], (args.seq_len - gt.shape[0], 1))))

        start_idx = 0
        end_idx = start_idx + args.seq_len
        ctr = 0
        while end_idx <= feat.shape[0]:
            save_path = args.data_dir + direc + '/' + vid_id + '_{}'.format(ctr)
            sio.savemat(save_path, {'relu6':feat[start_idx:end_idx,:], 'label':gt[start_idx:end_idx,:]})
            print 'Saved: {}'.format(save_path)
            start_idx += args.stride
            end_idx = start_idx + args.seq_len
            ctr += 1

def get_segments(y, delta=16):
    """Convert predicted output tensor (y_pred) from SST model into the
    corresponding temporal proposals. Can perform standard confidence
    thresholding/post-processing (e.g. non-maximum suppression) to select
    the top proposals afterwards.

    Parameters
    ----------
    y : ndarray
        Predicted output from SST model of size (L, K), where L is the length of
        the input video in terms of discrete time steps.
    delta : int, optional
        The temporal resolution of the visual encoder in terms of frames. See
        Section 3 of the main paper for additional details.

    Returns
    -------
    props : ndarray
        Two-dimensional array of shape (num_props, 2), containing the start and
        end boundaries of the temporal proposals in units of frames.
    scores : ndarray
        One-dimensional array of shape (num_props,), containing the
        corresponding scores for each detection above.
    """
    temp_props, temp_scores = [], []
    L, K = y.shape
    for i in range(L):
        for j in range(min(i+1, K)):
            temp_props.append([delta*(i-j-1), delta*i])
            temp_scores.append(y[i, j])
    props_arr, score_arr = np.array(temp_props), np.array(temp_scores)
    # filter out proposals that extend beyond the start of the video.
    idx_valid = props_arr[:, 0] >= 0
    props, scores = props_arr[idx_valid, :], score_arr[idx_valid]
    return props, scores


def nms_detections(props, scores, overlap=0.5):
    """Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously selected
    detection. This version is translated from Matlab code by Tomasz
    Malisiewicz, who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    props : ndarray
        Two-dimensional array of shape (num_props, 2), containing the start and
        end boundaries of the temporal proposals.
    scores : ndarray
        One-dimensional array of shape (num_props,), containing the corresponding
        scores for each detection above.

    Returns
    -------
    nms_props, nms_scores : ndarrays
        Arrays with the same number of dimensions as the original input, but
        with only the proposals selected after non-maximum suppression.
    """
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]
    return nms_props, nms_scores
