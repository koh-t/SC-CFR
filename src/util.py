# encoding: utf-8
# !/usr/bin/env python3

import sys
import pickle
import itertools
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class GetSeatImg():
    def __init__(self, colname):
        name = colname.to_list()
        name = [x.split('_')[1:] for x in name]
        self.name = np.array(name).astype(int) - 1

    def convert(self, _used):
        usedid = self.name[_used == 1]
        seatimg = np.zeros([22, 42])
        seatimg[usedid[:, 0], usedid[:, 1]] = 1
        seatimg = 1 - seatimg
        return seatimg


class Logger(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        self.log.close()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class DataSet:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index], self.z[index, :]


def pickle_dump(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj


def standard_scaler(x_train, x_test):
    _scaler = StandardScaler().fit(x_train)
    _x_train = _scaler.transform(x_train)
    _x_test = _scaler.transform(x_test)
    return _x_train, _x_test, _scaler


def minmax_scaler(x_train, x_test):
    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    _scaler = MinMaxScaler().fit(x_train)
    _x_train = _scaler.transform(x_train)
    _x_test = _scaler.transform(x_test)
    return _x_train, _x_test, _scaler


def load_data_conv(args, exp_id):
    X_seat = pd.read_csv('%s/covariate_seat_%d.csv' % (args.source_data_path, args.agent),
                         index_col=0)
    seat_column_name = X_seat.columns

    X_seat = X_seat.to_numpy()
    X_agent_within = pd.read_csv('%s/covariate_agent_within_%d.csv' % (args.source_data_path, args.agent),
                                 index_col=0).iloc[:, :6].to_numpy()
    X_prop = pickle_load(
        '%s/covariate_prop_%d.csv' % (args.source_data_path, args.agent))

    M = pickle_load(
        '%s/outcome_%d.csv' % (args.source_data_path, args.agent))
    Z = pickle_load(
        '%s/treatment_%d.csv' % (args.source_data_path, args.agent))

    factual_id = np.loadtxt(
        '%s/factural_%d_exp_%d.csv' % (args.biased_data_path, args.agent, exp_id))
    Y = pickle_load(
        '%s/outcome_withnoise_%d_exp_%d.csv' % (args.biased_data_path, args.agent, exp_id))

    # convert to img
    getseatimg = GetSeatImg(seat_column_name)
    X_seat_img = []
    for i in range(X_seat.shape[0]):
        X_seat_img.append(getseatimg.convert(X_seat[i, :]))
    X_seat_img = np.array(X_seat_img)

    id_train, id_test, X_seat_train, X_seat_test, X_seat_img_train, X_seat_img_test, X_agent_within_train, X_agent_within_test, X_prop_train, X_prop_test, Y_train, Y_test, M_train, M_test, Z_train, Z_test, factual_id_train, factual_id_test = train_test_split(
        np.arange(
            X_seat.shape[0]), X_seat, X_seat_img, X_agent_within, X_prop, Y, M, Z, factual_id,
        test_size=0.1,
        random_state=exp_id)

    # get factual treatment and outcome
    y_train = []
    z_train = []
    for (n, fid) in enumerate(factual_id_train):
        y_train.append(Y_train[n].iloc[int(fid), :])
        z_train.append(Z_train[n][int(fid), :])

    df_y_train = pd.DataFrame(y_train)
    y_train = df_y_train[args.outcome].to_numpy().reshape([-1, 1])
    z_train = np.array(z_train)

    return id_train, id_test,\
        X_seat_train, X_seat_test, X_seat_img_train, X_seat_img_test, X_agent_within_train, X_agent_within_test, X_prop_train, X_prop_test,\
        Y_train, Y_test, M_train, M_test, Z_train, Z_test, factual_id_train, factual_id_test,\
        y_train, z_train


def load_data(args, exp_id):
    X_seat = pd.read_csv('%s/covariate_seat_%d.csv' % (args.source_data_path, args.agent),
                         index_col=0).to_numpy()
    X_agent_within = pd.read_csv('%s/covariate_agent_within_%d.csv' % (args.source_data_path, args.agent),
                                 index_col=0).iloc[:, :6].to_numpy()
    X_prop = pickle_load(
        '%s/covariate_prop_%d.csv' % (args.source_data_path, args.agent))

    M = pickle_load(
        '%s/outcome_%d.csv' % (args.source_data_path, args.agent))
    Z = pickle_load(
        '%s/treatment_%d.csv' % (args.source_data_path, args.agent))

    factual_id = np.loadtxt(
        '%s/factural_%d_exp_%d.csv' % (args.biased_data_path, args.agent, exp_id))
    Y = pickle_load(
        '%s/outcome_withnoise_%d_exp_%d.csv' % (args.biased_data_path, args.agent, exp_id))

    id_train, id_test, X_seat_train, X_seat_test, X_agent_within_train, X_agent_within_test, X_prop_train, X_prop_test, Y_train, Y_test, M_train, M_test, Z_train, Z_test, factual_id_train, factual_id_test = train_test_split(
        np.arange(
            X_seat.shape[0]), X_seat, X_agent_within, X_prop, Y, M, Z, factual_id,
        test_size=0.1,
        random_state=exp_id)

    # get factual treatment and outcome
    y_train = []
    z_train = []
    for (n, fid) in enumerate(factual_id_train):
        y_train.append(Y_train[n].iloc[int(fid), :])
        z_train.append(Z_train[n][int(fid), :])

    df_y_train = pd.DataFrame(y_train)
    y_train = df_y_train[args.outcome].to_numpy()
    z_train = np.array(z_train)

    return id_train, id_test,\
        X_seat_train, X_seat_test, X_agent_within_train, X_agent_within_test, X_prop_train, X_prop_test,\
        Y_train, Y_test, M_train, M_test, Z_train, Z_test, factual_id_train, factual_id_test,\
        y_train, z_train


def get_pm_ate_pehe(model, x_test, Y_test, Z_test, target_outcome, ypred='', y_fact='', z_fact=''):
    N = len(x_test)
    if type(y_fact) != str:
        factual_mse = mean_squared_error(
            y_fact, model.predict(np.c_[x_test, z_fact]))

    combid = [list(x) for x in itertools.combinations(
        np.arange(Z_test[0].shape[0]), 2)]
    for (i, _x) in enumerate(x_test):
        if type(ypred) == str:
            _y = Y_test[i][target_outcome].to_numpy()
            _z = Z_test[i]
            _x = np.tile(_x, [_z.shape[0], 1])
            _ypred = model.predict(np.c_[_x, _z])
        else:
            _y = Y_test[i][target_outcome].to_numpy()
            Ntreat = Z_test[0].shape[0]
            s = Ntreat * i
            e = Ntreat * (i + 1)
            _ypred = ypred[s:e]

        # ATE, sum outcomes over population
        _m_comb = _y[combid]
        _ypred_comb = _ypred[combid]
        # PEHE, sum personal difference over population
        _m_pehe_comb = (_m_comb[:, 0] - _m_comb[:, 1])
        _ypred_pehe_comb = (_ypred_comb[:, 0] - _ypred_comb[:, 1])
        _pehe = np.power(_ypred_pehe_comb - _m_pehe_comb, 2)
        if i == 0:
            m_comb = _m_comb
            ypred_comb = _ypred_comb
            pehe = _pehe
            yt = _y
            ypredt = _ypred
        else:
            m_comb += _m_comb
            ypred_comb += _ypred_comb
            pehe += _pehe
            yt = np.r_[yt, _y]
            ypredt = np.r_[ypredt, _ypred]

    m_comb = m_comb / N
    ypred_comb = ypred_comb / N
    pehe = pehe / N
    mse = mean_squared_error(yt, ypredt)
    diff_abs_ate = np.abs(
        (ypred_comb[:, 0] - ypred_comb[:, 1]) - (m_comb[:, 0] - m_comb[:, 1]))

    pm_ate = np.mean(diff_abs_ate)
    pm_pehe = np.sqrt(np.mean(pehe))
    if type(y_fact) != str:
        return [pm_ate, pm_pehe, mse], factual_mse
    else:
        return [pm_ate, pm_pehe, mse]
