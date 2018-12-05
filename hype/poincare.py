#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch import nn  # noqa F401
from torch.autograd import Function
from .euclidean import expm, logm, ptransp, pnorm  # noqa F401

eps = 1e-5
boundary = 1 - eps
max_norm = boundary
spten_t = th.sparse.DoubleTensor


class Distance(Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v):
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, boundary)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, boundary)
        sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv


distance = Distance.apply


def rgrad(p, d_p):
    if d_p.is_sparse:
        p_sqnorm = th.sum(
            p[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        n_vals.renorm_(2, 0, 5)
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = th.sum(p ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def init_weights(w, scale=1e-4):
    w.data.uniform_(-scale, scale)
