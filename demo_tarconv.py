# encoding: utf-8
# !/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import torch

sys.path.append('./src/')  # noqa
from model import TARConv  # noqa
from util import Logger, pickle_dump, minmax_scaler, load_data_conv, DataSet  # noqa
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG


def extract_json(jfilepath):
    # JSONをチェック
    if not os.path.exists(jfilepath):
        js = {'len': 0, 'data': []}
        with open(jfilepath, 'w') as _f:
            json.dump(js, _f, indent=4)
    else:
        with open(jfilepath) as f:
            js = f.read()
        try:
            js = json.loads(js)
        except:
            decoder = json.JSONDecoder()
            js = decoder.raw_decode(js)[0]
    return js


def extract_csv(filepath):
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=['method', 'expid', 'train_rmse', 'valid_rmse',
                                   'without_rmse', 'without_pehe', 'without_ate'])
    else:
        df = pd.read_csv(filepath, index_col=0)
    return df


def check_json(args, js_result):
    js = {}
    js.update(vars(args))
    for _pop in ['din', 'dtreat', 'dout', 'device', 'disable_cuda', 'dirpath', 'log_dir']:
        try:
            js.pop(_pop)
        except:
            continue

    # 全部同じだったらプログラム終了
    for data in js_result['data']:
        flag = True
        for _key in js.keys():
            flag *= (data[_key] == js[_key])
        if flag == 1:
            sys.exit()


def transform_csv_js(filepath, jfilepath, result, args):
    # ------------------------------ #
    # 最新の結果を読み込む
    '''
    df = extract_csv(filepath)
    ser = pd.Series(result)
    # 実験結果を付け加える
    if len(df) != 0:
        df.append(pd.DataFrame(ser), ignore_index=True)
    else:
        df = pd.DataFrame(ser)
    '''
    # self.df = self.df.append({'method': self.args.model, 'expid': self.args.expid, 'train_rmse': train_rmse, 'valid_rmse': valid_rmse,
    #                           'without_rmse': rmse, 'without_pehe': pehe, 'without_ate': ate}, ignore_index=True)
    # ------------------------------ #

    # ------------------------------ #
    # 最新の結果を読み込む
    js = extract_json(jfilepath)
    # 結果を付け加える
    _js = result
    _js.update(vars(args))
    # js = {'method': self.args.model, 'expid': self.args.expid, 'train_rmse': round(train_rmse.item(), 3), 'valid_rmse': round(valid_rmse.item(), 3),
    #       'without_rmse': round(rmse.item(), 3), 'without_pehe': round(pehe.item(), 3), 'without_ate': round(ate.item(), 3)}
    for _pop in ['device', 'disable_cuda']:
        _js.pop(_pop)

    js['data'].append(_js)
    js['len'] = len(js['data'])
    # ------------------------------ #
    return js


def save(filepath, jfilepath, js):
    # 書き出し
    df = pd.DataFrame(js['data']).sort_values(by=['train-RMSE'])
    df.to_csv(filepath)

    with open(jfilepath, 'w+') as _f:
        json.dump(js, _f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TARConv')
    # Experiment setting
    parser.add_argument('--source_data_path', default='./data/source_data',
                        type=str, help='directory path of simulated data')
    parser.add_argument('--biased_data_path', default='./data/biased_data',
                        type=str, help='directory path of biased data')
    parser.add_argument('--outpath', type=str, default='./data/result')
    parser.add_argument('--agent', type=int, default=400,
                        help='set the number of agents in each evacuation')
    parser.add_argument('--outcome', type=str, default='MAX',
                        help='set the outcome from {MAX, MEAN, STD}')
    parser.add_argument('--expid', type=int, default=0,
                        help='set experiment id')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')

    # Model setting
    parser.add_argument('--model', type=str, default="TARConv")
    parser.add_argument('--hidden_rep', type=int, default=10)
    parser.add_argument('--c_rep', type=str, default='10,10')
    parser.add_argument('--c_out', type=str, default='80,50')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--sig', type=float, default=0.0)

    # Optimizer setting
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=1e-8)

    args = parser.parse_args()
    args.c_rep = [int(x) for x in args.c_rep.split(',')]
    args.c_out = [int(x) for x in args.c_out.split(',')]

    # CUDA setting
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # ------------------ #
    logger = getLogger(args.model)
    logger.setLevel(DEBUG)
    handler_format = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(handler_format)
    file_handler = FileHandler(
        args.outpath+'/logs/' + args.model+'-'+'{:%Y-%m-%d-%H:%M:%S}.log'.format(datetime.now()), 'a')
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug("Start process.")
    # ------------------ #

    # ------------------ #
    # check results
    outname = '%s/%s_%s_agent_%d.csv' % (
        args.outpath, args.model, args.outcome, args.agent)
    joutname = '%s/%s_%s_agent_%d.json' % (
        args.outpath, args.model, args.outcome, args.agent)
    js_result = extract_json(joutname)
    check_json(args, js_result)

    logger.debug('using data from %s and %s' %
                 (args.source_data_path, args.biased_data_path))

    logger.debug('save to %s' % args.outpath)
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
    # ------------------ #
    # sys.stdout = Logger(
    #     '%s/%s_result_%s_%d_exp_%d.txt' % (args.outpath, args.model, args.outcome, args.agent, args.expid))
    logger.debug('target outcome is %s. # of agents is %d. Experiment id is %d.' %
                 (args.outcome, args.agent, args.expid))

    # load data
    id_train, id_test,\
        X_seat_train, X_seat_test, X_seat_img_train, X_seat_img_test, X_agent_within_train, X_agent_within_test, X_prop_train, X_prop_test,\
        Y_train, Y_test, M_train, M_test, Z_train, Z_test, factual_id_train, factual_id_test,\
        y_train, z_train = load_data_conv(
            args, args.expid)

    # convert covariate
    x_train, x_test = X_seat_img_train, X_seat_img_test
    y_train, _, y_scaler = minmax_scaler(y_train, y_train)

    logger.debug(
        '# of samples = %d, # of features = [%d, %d]' % (x_train.shape))
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    z_train = torch.FloatTensor(z_train)

    dataset = DataSet(x_train, y_train, z_train)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=50, shuffle=True)

    din = x_train.shape[1]
    dtreat = z_train.shape[1]
    model = TARConv(din, dtreat, y_scaler, args).to(device=args.device)
    logger.debug(model)

    within_pm, outof_pm, train_mse = model.fit(
        dataloader, x_train, M_train, Z_train, x_test, M_test, Z_test, args.outcome, logger)
    del(model)

    result = {'within-RMSE': within_pm['RMSE'],
              'within-PEHE': within_pm['PEHE'],
              'within-ATE': within_pm['ATE'],
              'outof-RMSE': outof_pm['RMSE'],
              'outof-PEHE': outof_pm['PEHE'],
              'outof-ATE': outof_pm['ATE'],
              'train-RMSE': train_mse[-1]}

    csv_result = extract_csv(outname)
    js_result = extract_json(joutname)

    dict_args = vars(args)
    for key in dict_args.keys():
        result[key] = dict_args[key]
    df_result = pd.DataFrame.from_dict(result, orient='Index').T
    df_result = df_result.sort_values('within-RMSE')

    logger.debug('----- Result -----')
    logger.debug('Model name:\n%s' % args.model)
    logger.debug('Scores:')
    logger.debug('\t\twithin sample, \t\t\tout of sample')
    logger.debug(
        'Train MSE, \t[RMSE, PEHE, ATE], \t[RMSE, PEHE, ATE]')
    logger.debug('[%.3f], \t [%.3f, %.3f, %.3f], \t [%.3f, %.3f, %.3f]' % (result['train-RMSE'],
                                                                           result['within-RMSE'],
                                                                           result['within-PEHE'],
                                                                           result['within-ATE'],
                                                                           result['outof-RMSE'],
                                                                           result['outof-PEHE'],
                                                                           result['outof-ATE'],
                                                                           ))

    logger.debug('----- -----')
    js = transform_csv_js(outname, joutname, result, args)
    save(outname, joutname, js)

    logger.debug(0)
