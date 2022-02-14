# encoding: utf-8
# !/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def pickle_dump(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dataset Generator that uses outputs of a multi-agent simulator')
    parser.add_argument('--dirpath', default='./data/source_data',
                        type=str, help='directory path of source data')
    parser.add_argument('--outpath', default='./data/biased_data',
                        type=str, help='directory path of generated data')
    parser.add_argument('--agent', default=400, type=int,
                        help='minimumn number of agents')
    parser.add_argument('--Nexp', default=10, type=int,
                        help='number of data sets')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='optional parameter for the route guide using a distribution')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='optional parameter for the route guide using a distribution')

    args = parser.parse_args()

    print('agent>=%d' % args.agent)

    print('save to %s' % args.outpath)
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    # -------------------- #
    # load preprocessed data
    # -------------------- #
    # agent locations
    X_seat = pd.read_csv('%s/covariate_seat_%d.csv' %
                         (args.dirpath, args.agent), index_col=0)
    # seat within the range from doors
    X_agent_within = pd.read_csv('%s/covariate_agent_within_%d.csv' % (args.dirpath, args.agent),
                                 index_col=0).iloc[:, :6].to_numpy()
    # actual evacuation time
    M = pickle_load('%s/outcome_%d.csv' %
                    (args.dirpath, args.agent))
    # treatment information
    Z = pickle_load('%s/treatment_%d.csv' %
                    (args.dirpath, args.agent))

    for exp_id in range(args.Nexp):
        # select factual treatment

        # treatment for guide guide
        _N_agent = X_seat.sum(1).to_numpy().reshape([-1, 1])
        scaler = StandardScaler().fit(_N_agent)
        N_agent = scaler.transform(_N_agent)
        propensity = 1 / (1 + np.exp(-N_agent * args.alpha + args.beta))
        # generate treatment by a random sampling
        odds = np.random.rand(N_agent.shape[0], 1)
        z_guide = (odds < propensity).astype(float)

        # treatment for door operations
        rank_within_door = np.argsort(X_agent_within, 1)
        P_agent_within = X_agent_within / X_agent_within.sum(1).reshape(-1, 1)
        # generate treatment by a random sampling
        z_door_open = np.zeros([P_agent_within.shape[0], 2])
        for n in range(P_agent_within.shape[0]):
            z_door_open[n, :] = np.random.choice(
                6, 2, p=P_agent_within[n, :], replace=False)
        z_door_open = z_door_open.astype(int)+1
        print('door index and opened counts', np.unique(
            z_door_open.reshape([-1]), return_counts=True))

        # get factual treatment
        zid = []
        for n in range(z_guide.shape[0]):
            zid.append([int(z_guide[n][0]), z_door_open[n]
                       [0], z_door_open[n][1]])

        factual_id = []
        for (n, _zid) in enumerate(zid):
            for (j, __zid) in enumerate(_zid):
                if j == 0:
                    id = Z[n][:, j] == __zid
                else:
                    id = id & (Z[n][:, __zid] == 1)
            factual_id.append(np.where(id == True)[0][0])

        factual_id = np.array(factual_id)
        np.savetxt('%s/factural_%d_exp_%d.csv' %
                   (args.outpath, args.agent, exp_id), factual_id)

        # generate outcomes by adding a gaussian noise to the actual evacuation time
        Y = M.copy()
        for i in range(len(Y)):
            Y[i] = Y[i] + np.random.randn(Y[i].shape[0], Y[i].shape[1]) * 2

        pickle_dump(
            '%s/outcome_withnoise_%d_exp_%d.csv' % (args.outpath, args.agent, exp_id), Y)

    print('finish')
