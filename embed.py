#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import numpy as np
import logging
import argparse
from hype.sn import Embedding, initialize
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype import train
from hype.graph import load_adjacency_matrix, load_edge_list, eval_reconstruction
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
from hype.arsgd import RiemannianAveragedSGD
from hype.common import log_bistro
import sys
import json

th.manual_seed(42)
np.random.seed(42)


if __name__ == '__main__':  # noqa C901
    parser = argparse.ArgumentParser(description='Train Hyperbolic Embeddings')
    parser.add_argument('-checkpoint', default='/tmp/hype_embeddings.pth',
                        help='Where to store the model checkpoint')
    parser.add_argument('-dset', type=str, required=True,
                        help='Dataset identifier')
    parser.add_argument('-dim', type=int, default=20,
                        help='Embedding dimension')
    parser.add_argument('-manifold', type=str, default='lorentz'
                        , help='Embedding manifold')
    parser.add_argument('-lr', type=float, default=1000,
                        help='Learning rate')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-batchsize', type=int, default=12800,
                        help='Batchsize')
    parser.add_argument('-negs', type=int, default=50,
                        help='Number of negatives')
    parser.add_argument('-burnin', type=int, default=20,
                        help='Epochs of burn in')
    parser.add_argument('-dampening', type=float, default=0.75,
                        help='Sample dampening during burnin')
    parser.add_argument('-ndproc', type=int, default=8,
                        help='Number of data loading processes')
    parser.add_argument('-eval_each', type=int, default=1,
                        help='Run evaluation every n-th epoch')
    parser.add_argument('-fresh', action='store_true', default=False,
                        help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int,
                        help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-asgd', action='store_true', default=False,
                        help='Train with Riemannian Averaged SGD')
    parser.add_argument('-sym', action='store_true', default=False,
                        help='Symmetrize dataset')
    parser.add_argument('-maxnorm', default='500000', type=int)
    parser.add_argument('-sparse', default=False, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-reconstruction', action='store_true',
                        help='enable reconstruction evaluation')
    parser.add_argument('-burnin_multiplier', default=0.01, type=float)
    parser.add_argument('-neg_multiplier', default=1.0, type=float)
    opt = parser.parse_args()

    # set default tensor type
    th.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = th.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 else 'cpu')

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('lorentz')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    # select manifold to optimize on
    if opt.manifold == 'lorentz':
        from hype import lorentz as manifold
        opt.dim = opt.dim + 1
    elif opt.manifold == 'poincare':
        from hype import poincare as manifold
    elif opt.manifold == 'euclidean':
        from hype import euclidean as manifold
    else:
        raise ValueError(f'Unknown manifold {opt.manifold}')
    manifold.debug = opt.debug

    if 'csv' in opt.dset:
        log.info('Using edge list dataloader')
        idx, objects, weights = load_edge_list(opt.dset, opt.sym)
        model, data, model_name, conf = initialize(
            manifold, opt, idx, objects, weights, sparse=opt.sparse
        )
    else:
        log.info('Using adjacency matrix dataloader')
        dset = load_adjacency_matrix(opt.dset, 'hdf5')
        log.info('Setting up dataset...')
        data = AdjacencyDataset(dset, opt.negs, opt.batchsize, opt.ndproc,
            opt.burnin > 0)
        model = Embedding(data.N, opt.dim, manifold, sparse=opt.sparse)
        objects = dset['objects']

    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    train._lr_multiplier = opt.burnin_multiplier

    # setup optimizer
    oparams = model.optim_params(manifold)
    if opt.asgd:
        model.w_avg = model.lt.weight.data.clone().to(device)
        optimizer = RiemannianAveragedSGD(
            oparams,
            params_avg=[model.w_avg],
            lr=opt.lr,
            t0=2 * opt.burnin
        )
    else:
        optimizer = RiemannianSGD(model.optim_params(manifold), lr=opt.lr)

    # Build config string for log
    log.info(f'json_conf: {json.dumps(vars(opt))}')

    # setup checkpoint
    checkpoint = LocalCheckpoint(
        opt.checkpoint,
        include_in_all={'conf' : vars(opt), 'objects' : objects},
        start_fresh=opt.fresh
    )

    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']

    def checkpointer(model, epoch, loss):
        lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data
        checkpoint.save({
            'model': model.state_dict(),
            'embeddings': lt.cpu().numpy(),
            'epoch': epoch,
            'manifold': manifold.__name__,
        })

    if opt.reconstruction:
        adj = {}
        for inputs, _ in data:
            for row in inputs:
                x = row[0].item()
                y = row[1].item()
                if x in adj:
                    adj[x].add(y)
                else:
                    adj[x] = {y}

    # control closure
    def control(model, epoch, elapsed, loss):
        """
        Control thread to evaluate embedding
        """
        lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data
        sqnorms = manifold.pnorm(lt)
        lmsg = {
            'epoch': (epoch, 'd'),
            'elapsed': (elapsed, '.2f'),
            'loss': (loss, '.3f'),
            'sqnorm_min': (sqnorms.min().item(), '.7f'),
            'sqnorm_avg': (sqnorms.mean().item(), '.7f'),
            'sqnorm_max': (sqnorms.max().item(), '.7f'),
        }

        if opt.reconstruction:
            mean_rank, map_rank = eval_reconstruction(adj, lt, manifold.distance)
            lmsg.update({
                'mean_rank': (mean_rank, '.2f'),
                'map_rank': (map_rank, '.2f'),
            })

        log_bistro(log, lmsg)

    control.checkpoint = True
    model = model.to(device)
    if hasattr(model, 'w_avg'):
        model.w_avg = model.w_avg.to(device)

    train.train(device, model, data, optimizer, opt, log, ctrl=control,
        checkpointer=checkpointer, progress=True)
