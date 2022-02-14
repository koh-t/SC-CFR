# encoding: utf-8
# !/usr/bin/env python3

import itertools
import numpy as np
from matplotlib import pylab as plt

import torch
import torch.nn as nn
from torch import optim
from layer import DMLLinear, MLP, ConvNet, DMLConvNet
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error


def mmd_rbf(Xt, Xc, p, sig=0.1):
    sig = torch.tensor(sig)
    Kcc = torch.exp(-torch.cdist(Xc, Xc, 2.0001) / torch.sqrt(sig))
    Kct = torch.exp(-torch.cdist(Xc, Xt, 2.0001) / torch.sqrt(sig))
    Ktt = torch.exp(-torch.cdist(Xt, Xt, 2.0001) / torch.sqrt(sig))

    m = Xc.shape[0]
    n = Xt.shape[0]

    mmd = (1 - p) ** 2 / (m * m) * (Kcc.sum() - m)
    mmd += p ** 2 / (n * n) * (Ktt.sum() - n)
    mmd -= - 2 * p * (1 - p) / (m * n) * Kct.sum()
    mmd *= 4
    '''
    mmd = (1 - p) ** 2 / (m * (m - 1)) * (Kcc.sum() - m)
    mmd += p ** 2 / (n * (n - 1)) * (Ktt.sum() - n)
    mmd -= - 2 * p * (1 - p) / (m * n) * Kct.sum()
    mmd *= 4
    '''
    return mmd


class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss()
        self.mse = mean_squared_error

    def fit(self, dataloader, x_train, M_train, Z_train, x_test, M_test, Z_test, target_outcome):
        losses = []
        print('                        within sample,      out of sample')
        print(
            '           [Train MSE], [RMSE, PEHE, ATE], [RMSE, PEHE, ATE]')
        for epoch in range(self.args.epoch):
            epoch_loss = 0
            n = 0
            for (x, y, z) in dataloader:
                if self.args.alpha > 0.0:
                    zuniq, zcount = np.unique(
                        z.cpu().detach().numpy(), axis=0, return_counts=True)

                x = x.to(device=self.args.device)
                y = y.to(device=self.args.device)
                z = z.to(device=self.args.device)
                self.optimizer.zero_grad()
                y_hat = self.forward(x, y, z)
                loss = self.criterion(y_hat, y.reshape([-1, 1]))

                mse = self.mse(self.y_scaler.inverse_transform(y_hat.detach().cpu().numpy()),
                               self.y_scaler.inverse_transform(y.reshape([-1, 1]).detach().cpu().numpy()))
                loss.backward()

                self.optimizer.step()
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]

            self.scheduler.step()
            epoch_loss = epoch_loss / n
            losses.append(epoch_loss)

            if epoch % 10 == 0:
                with torch.no_grad():
                    # ATE, sqrt(pehe), cmse
                    within_pm = get_score(
                        self, x_train, M_train, Z_train, target_outcome)
                    outof_pm = get_score(
                        self, x_test, M_test, Z_test, target_outcome)
                print('[Epoch: %d] [%.3f], [%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f] ' %
                      (epoch, epoch_loss,
                       within_pm['RMSE'], within_pm['PEHE'], within_pm['ATE'],
                       outof_pm['RMSE'], outof_pm['PEHE'], outof_pm['ATE']))

        return within_pm, outof_pm, losses

    def get_score(self, model, x_test, y_test, z_test, target_outcome):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        N = len(x_test)
        combid = [list(x) for x in itertools.combinations(
            np.arange(z_test[0].shape[0]), 2)]
        for (i, _x) in enumerate(x_test):
            _y = y_test[i][target_outcome].to_numpy().reshape([-1, 1])
            _z = z_test[i]

            if model._get_name() in ['TARConv', 'TARConvGCN', 'CFRConv', 'DML', 'DMLShared']:
                _x = np.tile(_x, [_z.shape[0], 1, 1])
            else:
                _x = np.tile(_x, [_z.shape[0], 1])
            _x = torch.FloatTensor(_x).to(device=device)
            _z = torch.FloatTensor(_z).to(device=device)

            _ypred = model.forward(_x, _z)
            _ypred = self.y_scaler.inverse_transform(
                _ypred.detach().cpu().numpy())
            if type(_ypred) != np.ndarray:
                _ypred = _ypred.detach().cpu().numpy()

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

        if type(diff_abs_ate) != np.ndarray:
            diff_abs_ate = diff_abs_ate.numpy()
            pehe = pehe.numpy()

        pm_ate = np.mean(diff_abs_ate)
        pm_pehe = np.mean(pehe)
        return {'ATE': pm_ate, 'PEHE': np.sqrt(pm_pehe), 'RMSE': np.sqrt(mse)}


class DML(Base):
    def __init__(self, din, dtreat, y_scaler, args):
        super().__init__(args)
        self.xnet = DMLConvNet(din=din, dout=args.hidden_rep, C=[
            args.c_rep[0], args.c_rep[1]])
        self.repnet = ConvNet(din=din, dout=args.hidden_rep, C=[
            args.c_rep[0], args.c_rep[1]])
        self.outnet = MLP(din=3 * args.hidden_rep + dtreat,
                          dout=1, C=[args.c_out[0], args.c_out[1]])
        self.params = list(self.repnet.parameters()) + \
            list(self.outnet.parameters())
        self.optimizer = optim.Adam(
            params=self.params, lr=args.lr, weight_decay=args.wd)
        self.scheduler = StepLR(self.optimizer, step_size=args.step, gamma=0.1)
        self.y_scaler = y_scaler

    def forward(self, x, z):
        y_hat_x, _ = self.xnet(x)

        _, x_rep_stack = self.repnet(x)
        x_rep = x_rep_stack[2]
        y_hat = self.outnet(torch.cat((x_rep, z), 1))

        return y_hat_x + y_hat

    def fit(self, dataloader, x_train, M_train, Z_train, x_test, M_test, Z_test, target_outcome):
        losses = []
        print('                        within sample,      out of sample')
        print(
            '           [Train MSE], [RMSE, PEHE, ATE], [RMSE, PEHE, ATE]')
        for epoch in range(self.args.epoch):
            epoch_loss = 0
            n = 0
            for (x, y, z) in dataloader:
                x = x.to(device=self.args.device)
                y = y.to(device=self.args.device)
                z = z.to(device=self.args.device)
                self.optimizer.zero_grad()
                y_hat = self.forward(x, z)
                loss = self.criterion(y_hat, y.reshape([-1, 1]))
                loss.backward()

                mse = self.mse(self.y_scaler.inverse_transform(y_hat.detach().cpu().numpy()),
                               self.y_scaler.inverse_transform(y.reshape([-1, 1]).detach().cpu().numpy()))
                self.optimizer.step()
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]

            self.scheduler.step()
            epoch_loss = epoch_loss / n
            losses.append(epoch_loss)

            if epoch % 10 == 0:
                with torch.no_grad():
                    # ATE, sqrt(pehe), cmse
                    within_pm = self.get_score(
                        self, x_train, M_train, Z_train, target_outcome)
                    outof_pm = self.get_score(
                        self, x_test, M_test, Z_test, target_outcome)
                print('[Epoch: %d] [%.3f], [%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f] ' %
                      (epoch, epoch_loss,
                       within_pm['RMSE'], within_pm['PEHE'], within_pm['ATE'],
                       outof_pm['RMSE'], outof_pm['PEHE'], outof_pm['ATE']))

        return within_pm, outof_pm, losses


class DMLShared(Base):
    def __init__(self, din, dtreat, y_scaler, args):
        super().__init__(args)
        self.xnet = DMLLinear(din=3*args.hidden_rep)
        self.repnet = ConvNet(din=din, dout=args.hidden_rep, C=[
            args.c_rep[0], args.c_rep[1]])
        self.outnet = MLP(din=3 * args.hidden_rep + dtreat,
                          dout=1, C=[args.c_out[0], args.c_out[1]])
        self.params = list(self.repnet.parameters()) + \
            list(self.outnet.parameters())
        self.optimizer = optim.Adam(
            params=self.params, lr=args.lr, weight_decay=args.wd)
        self.scheduler = StepLR(self.optimizer, step_size=args.step, gamma=0.1)
        self.y_scaler = y_scaler

    def forward(self, x, z):
        _, x_rep_stack = self.repnet(x)
        x_rep = x_rep_stack[2]
        y_hat_x = self.xnet(x_rep)

        _, x_rep_stack = self.repnet(x)
        x_rep = x_rep_stack[2]
        y_hat = self.outnet(torch.cat((x_rep, z), 1))

        return y_hat_x + y_hat

    def fit(self, dataloader, x_train, M_train, Z_train, x_test, M_test, Z_test, target_outcome):
        losses = []
        print('                        within sample,      out of sample')
        print(
            '           [Train MSE], [RMSE, PEHE, ATE], [RMSE, PEHE, ATE]')
        for epoch in range(self.args.epoch):
            epoch_loss = 0
            n = 0
            for (x, y, z) in dataloader:
                x = x.to(device=self.args.device)
                y = y.to(device=self.args.device)
                z = z.to(device=self.args.device)
                self.optimizer.zero_grad()
                y_hat = self.forward(x, z)
                loss = self.criterion(y_hat, y.reshape([-1, 1]))
                loss.backward()

                mse = self.mse(self.y_scaler.inverse_transform(y_hat.detach().cpu().numpy()),
                               self.y_scaler.inverse_transform(y.reshape([-1, 1]).detach().cpu().numpy()))
                self.optimizer.step()
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]

            self.scheduler.step()
            epoch_loss = epoch_loss / n
            losses.append(epoch_loss)

            if epoch % 10 == 0:
                with torch.no_grad():
                    # ATE, sqrt(pehe), cmse
                    within_pm = self.get_score(
                        self, x_train, M_train, Z_train, target_outcome)
                    outof_pm = self.get_score(
                        self, x_test, M_test, Z_test, target_outcome)
                print('[Epoch: %d] [%.3f], [%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f] ' %
                      (epoch, epoch_loss,
                       within_pm['RMSE'], within_pm['PEHE'], within_pm['ATE'],
                       outof_pm['RMSE'], outof_pm['PEHE'], outof_pm['ATE']))

        return within_pm, outof_pm, losses
