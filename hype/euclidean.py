#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch import nn

max_norm = 1


class Distance(nn.Module):
    def __init__(self, radius=1, dim=None):
        super(Distance, self).__init__()

    def forward(self, u, v):
        return th.sum(th.pow(u - v, 2), dim=-1)


def pnorm(u, dim=-1):
    return th.sqrt(th.sum(u * u, dim=dim))


def init_weights(w, scale=1e-4):
    w.uniform_(-scale, scale)


def rgrad(p, d_p):
    return d_p


def expm(p, d_p, lr=None, out=None):
    if lr is not None:
        d_p.mul_(-lr)
    if out is None:
        out = p
    out.add_(d_p)
    return out


def logm(p, d_p, out=None):
    return p - d_p


def ptransp(p, x, y, v):
    ix, v_ = v._indices().squeeze(), v._values()
    return p.index_copy_(0, ix, v_)


class TranseDistance(nn.Module):
    def __init__(self, radius=1, dim=None):
        super(TranseDistance, self).__init__()
        self.r = nn.Parameter(th.randn(dim).view(1, dim))

    def forward(self, u, v):
        # batch mode
        if u.dim() == 3:
            r = self.r.unsqueeze(0).expand(v.size(0), v.size(1), self.r.size(1))
        # non batch
        else:
            r = self.r.expand(v.size(0), self.r.size(1))
        return th.sum(th.pow(u - v + r, 2), dim=-1)
