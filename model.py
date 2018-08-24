#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from numpy.random import choice, randint
import torch as th
from torch import nn
from torch.autograd import Function, Variable
from torch.utils.data import Dataset
from collections import defaultdict as ddict

eps = 1e-5


class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = th.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


class PoincareDistance(Function):
    boundary = 1 - eps
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
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, PoincareDistance.boundary)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, PoincareDistance.boundary)
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
        gu = PoincareDistance.grad(u, v, squnorm, sqvnorm, sqdist)
        gv = PoincareDistance.grad(v, u, sqvnorm, squnorm, sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

poincare_distance = PoincareDistance.apply

class EuclideanDistance(nn.Module):
    def __init__(self, radius=1, dim=None):
        super(EuclideanDistance, self).__init__()

    def forward(self, u, v):
        return th.sum(th.pow(u - v, 2), dim=-1)


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


class Embedding(nn.Module):
    def __init__(self, size, dim, dist=poincare_distance, max_norm=1):
        super(Embedding, self).__init__()
        self.dim = dim
        self.lt = nn.Embedding(
            size, dim,
            max_norm=max_norm,
            sparse=True,
            scale_grad_by_freq=False
        )
        self.dist = dist
        self.init_weights()

    def init_weights(self, scale=1e-4):
        self.lt.state_dict()['weight'].uniform_(-scale, scale)

    def forward(self, inputs):
        e = self.lt(inputs)
        fval = self._forward(e)
        return fval

    def embedding(self):
        return list(self.lt.parameters())[0].data.cpu().numpy()


class SNEmbedding(Embedding):
    def __init__(self, size, dim, dist=poincare_distance, max_norm=1):
        super(SNEmbedding, self).__init__(size, dim, dist, max_norm)
        self.lossfn = nn.CrossEntropyLoss

    def _forward(self, e):
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.dist(s, o).squeeze(-1)
        return -dists

    def loss(self, preds, targets, weight=None):
        lossfn = self.lossfn(reduction='elementwise_mean', weight=weight)
        return lossfn(preds, targets)


class GraphDataset(Dataset):
    _ntries = 10
    _dampening = 1

    def __init__(self, idx, objects, nnegs, unigram_size=1e8):
        print('Indexing data')
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects
        self.max_tries = self.nnegs * self._ntries

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(len(objects), dtype=np.float)
        for i in range(idx.size(0)):
            t, h, w = self.idx[i]
            self._counts[h.item()] += w
            self._weights[t.item()][h.item()] += w
        self._weights = dict(self._weights)
        nents = int(np.array(list(self._weights.keys())).max() + 1)
        assert len(objects) == nents, 'Number of objects do no match'

        if unigram_size > 0:
            c = self._counts ** self._dampening
            self.unigram_table = choice(
                len(objects),
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.size(0)

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return Variable(th.cat(inputs, 0)), Variable(th.cat(targets, 0))


class SNGraphDataset(GraphDataset):
    model_name = '%s_%s_dim%d'

    def __getitem__(self, i):
        t, h, _ = self.idx[i]
        negs = set()
        ntries = 0
        nnegs = self.nnegs
        if self.burnin:
            nnegs *= 0.1
        while ntries < self.max_tries and len(negs) < nnegs:
            if self.burnin:
                n = randint(0, len(self.unigram_table))
                n = int(self.unigram_table[n])
            else:
                n = randint(0, len(self.objects))
            if n not in self._weights[t.item()]:
                negs.add(n)
            ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()

    @classmethod
    def initialize(cls, distfn, opt, idx, objects, max_norm=1):
        conf = []
        model_name = cls.model_name % (opt.dset, opt.distfn, opt.dim)
        data = cls(idx, objects, opt.negs)
        model = SNEmbedding(
            len(data.objects),
            opt.dim,
            dist=distfn,
            max_norm=max_norm
        )
        data.objects = objects
        return model, data, model_name, conf
