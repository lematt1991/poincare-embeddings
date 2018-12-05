#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.autograd import Function
from .common import acosh

eps = 1e-12
_eps = 1e-5
norm_clip = 1
debug = False
max_norm = None


def ldot(u, v, keepdim=False):
    """Lorentzian Scalar Product"""
    uv = u * v
    uv.narrow(-1, 0, 1).mul_(-1)
    return th.sum(uv, dim=-1, keepdim=keepdim)


def to_poincare_ball(u):
    x = u.clone()
    d = x.size(-1) - 1
    return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)


class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u


def distance(u, v):
    d = -LorentzDot.apply(u, v)
    d.data.clamp_(min=1)
    return acosh(d, _eps)


def pnorm(u):
    return th.sqrt(th.sum(th.pow(to_poincare_ball(u), 2), dim=-1))


def normalize(w):
    """Normalize vector such that it is located on the hyperboloid"""
    d = w.size(1) - 1
    narrowed = w.narrow(1, 1, d)
    if max_norm:
        narrowed.renorm_(2, 0, max_norm)
    tmp = 1 + th.sum(th.pow(narrowed, 2), dim=1, keepdim=True)
    tmp.sqrt_()
    w.narrow(1, 0, 1).copy_(tmp)
    return w


def normalize_tan(x_all, v_all):
    d = v_all.size(1) - 1
    x = x_all.narrow(1, 1, d)
    xv = th.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
    tmp = 1 + th.sum(th.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
    tmp.sqrt_().clamp_(min=_eps)
    v_all.narrow(1, 0, 1).copy_(xv / tmp)
    return v_all


def init_weights(w, irange=1e-5, unit_norm=False):
    w.data.uniform_(-irange, irange)
    w.data.copy_(normalize(w.data))


def rgrad(p, d_p):
    """Riemannian gradient for hyperboloid"""
    if d_p.is_sparse:
        u = d_p._values()
        x = p.index_select(0, d_p._indices().squeeze())
    else:
        u = d_p
        x = p
    u.narrow(-1, 0, 1).mul_(-1)
    u.addcmul_(ldot(x, u, keepdim=True).expand_as(x), x)
    return d_p


def expm(p, d_p, lr=None, out=None):
    """Exponential map for hyperboloid"""
    if out is None:
        out = p
    if d_p.is_sparse:
        ix, d_val = d_p._indices().squeeze(), d_p._values()
        p_val = p.index_select(0, ix)
        ldv = ldot(d_val, d_val, keepdim=True)
        if debug:
            assert all(ldv > 0), "Tangent norm must be greater 0"
            assert all(ldv == ldv), "Tangent norm includes NaNs"
        nd_p = ldv.clamp_(min=0).sqrt_()
        t = th.clamp(nd_p, max=norm_clip)
        nd_p.clamp_(min=eps)
        p.index_copy_(
            0, ix,
            normalize((th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p))
        )
    else:
        if lr is not None:
            d_p.narrow(-1, 0, 1).mul_(-1)
            d_p.addcmul_((ldot(p, d_p, keepdim=True)).expand_as(p), p)
            d_p.mul_(-lr)
        ldv = ldot(d_p, d_p, keepdim=True)
        if debug:
            assert all(ldv > 0), "Tangent norm must be greater 0"
            assert all(ldv == ldv), "Tangent norm includes NaNs"
        nd_p = ldv.clamp_(min=0).sqrt_()
        t = th.clamp(nd_p, max=norm_clip)
        nd_p.clamp_(min=eps)
        p.copy_(
            normalize((th.cosh(t) * p).addcdiv_(th.sinh(t) * d_p, nd_p))
        )


def logm(x, y):
    """Logarithmic map on the Lorenz Manifold"""
    xy = th.clamp(ldot(x, y).unsqueeze(-1), max=-1)
    v = acosh(-xy, eps).div_(
        th.clamp(th.sqrt(xy * xy - 1), min=_eps)
    ) * th.addcmul(y, xy, x)
    return normalize_tan(x, v)


def ptransp(x, y, v, ix=None, out=None):
    """Parallel transport for hyperboloid"""
    if ix is not None:
        v_ = v
        x_ = x.index_select(0, ix)
        y_ = y.index_select(0, ix)
    elif v.is_sparse:
        ix, v_ = v._indices().squeeze(), v._values()
        x_ = x.index_select(0, ix)
        y_ = y.index_select(0, ix)
    else:
        raise NotImplementedError("Lazy max")
    xy = ldot(x_, y_, keepdim=True).expand_as(x_)
    vy = ldot(v_, y_, keepdim=True).expand_as(x_)
    vnew = v_ + vy / (1 - xy) * (x_ + y_)
    if out is None:
        return vnew
    else:
        out.index_copy_(0, ix, vnew)
