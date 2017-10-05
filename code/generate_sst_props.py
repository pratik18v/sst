"""
generate_sst_props.py
---------------------
This is an example script to load pre-trained model parameters for SST and
obtain predictions on a set of videos. Note that this script is operating for
demo purposes on top of the visual encoder features for each time step in the
input videos.
"""
import argparse
import os

import hickle as hkl
import lasagne
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import scipy.io as sio

from sst.vis_encoder import VisualEncoderFeatures as VEFeats
from sst.model import SSTSequenceEncoder
from sst.utils import get_segments, nms_detections

def parse_args():
    p = argparse.ArgumentParser(
      description="SST example evaluation script",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-nms', '--nms-thresh', default=0.7, type=float,
                   help='threshold for non-maximum suppression')

    p.add_argument('-mp', '--model-params', help='filepath to model params file.',
                   default='../data/params/params_sl128_np32_d1_w256_fd500.hkl', type=str)

    p.add_argument('-od', '--output-dir', default='../data/proposals/output',
                   help='folder filepath for output proposals', type=str)

    p.add_argument('-d', '--dataset',
            default='/nfs/bigbang/pratik18v/cse599/sst/data/c3d_pca/',
             help='filepath for test dataset directory', type=str)

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

    p.add_argument('-drp', '--dropout', default=0.5,
            help='Dropout probability', type=float)

    p.add_argument('-v', '--verbose', default=False,
                   help='filename for output proposals', type=bool)

    return p.parse_args()

def load_model(input_var=None, target_var=None, args=None,  **kwargs):
    model = SSTSequenceEncoder(input_var, target_var, seq_length=args.seq_length, depth=args.depth,
        width=args.width, num_proposals=args.num_proposals, input_size=args.feat_dim, dropout=args.dropout,
        mode='test')
    return model


def main(args):
    # build the model network and load with pre-trained parameters
    input_var = T.tensor3('inputs')
    sst_model = load_model(input_var=input_var, args=args)
    sst_model.compile()
    sst_model.load_model_params(args.model_params)

    #Listing data files
    fnames = []
    video_ids = []
    for fname in os.listdir(args.dataset):
        if fname.split('_')[1] == 'test':
            fnames.append(args.dataset + fname)
            video_ids.append(fname.split('.')[0])

    n_vid = len(video_ids)
    proposals = [None] * n_vid
    video_name = [None] * n_vid

    for i, vid_name in enumerate(video_ids):
        # process each video stream individually
        data = sio.loadmat(fnames[i])
        X_t = np.expand_dims(data['relu6'], axis=0)
        # obtain proposals
        y_pred = sst_model.forward_eval(X_t)
        props_raw, scores_raw = get_segments(y_pred[0, :, :])
        props, scores = nms_detections(props_raw, scores_raw, args.nms_thresh)
        n_prop_after_pruning = scores.size

        proposals[i] = np.hstack([
            props, scores.reshape((-1, 1)),
            np.zeros((n_prop_after_pruning, 1))])
        video_name[i] = np.repeat([vid_name], n_prop_after_pruning).reshape(
            n_prop_after_pruning, 1)

    proposals_arr = np.vstack(proposals)
    proposals_vid = np.vstack(video_name)
    output_name = 'results_k{}.csv'.format(args.num_proposals)
    output_file = os.path.join(args.output_dir, output_name)
    df = pd.concat([
        pd.DataFrame(proposals_arr, columns=['f-init', 'f-end', 'score',
                                             'video-frames']),
        pd.DataFrame(proposals_vid, columns=['video-name'])],
        axis=1)
    df.to_csv(output_file, index=None, sep=' ')
    if args.verbose:
        print('successful execution')
    return 0

if __name__ == '__main__':
    args = parse_args()
    main(args)
