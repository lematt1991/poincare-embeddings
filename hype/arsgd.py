#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.optim.optimizer import Optimizer, required


class RiemannianAveragedSGD(Optimizer):
    r"""Riemannian averaged stochastic gradient descent.

    Args:
        rgrad (Function): Function to compute the Riemannian gradient
           from the Euclidean gradient
        retraction (Function): Function to update the retraction
           of the Riemannian gradient
    """

    def __init__(
            self,
            params,
            params_avg,
            lr=required,
            rgrad=required,
            expm=required,
            logm=required,
            t0=required,
    ):
        defaults = {
            'lr': lr,
            'rgrad': rgrad,
            'expm': expm,
            'logm': logm,
            't0': t0,
        }
        super(RiemannianAveragedSGD, self).__init__(params, defaults)
        for group in self.param_groups:
            group['params_avg'] = list(params_avg)
            group['counts'] = params_avg[0].new_zeros(params_avg[0].size(0), 1)

    def step(self, lr=None, counts=None):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']
                logm = group['logm']
                p_avg = group['params_avg'][0]

                if p.grad is None:
                    continue
                if counts is None:
                    group['counts'] += 1
                else:
                    group['counts'] += counts
                d_p = p.grad.data
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = rgrad(p.data, d_p)
                d_p.mul_(-lr)
                expm(p.data, d_p)
                t = th.clamp(group['counts'] - group['t0'], min=1)
                expm(p_avg, logm(p_avg, p.data) / t)

        return loss
