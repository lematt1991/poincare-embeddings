---
id: rsgd
title: Module hype.rsgd
sidebar_label: Module hype.rsgd
---
Classes
-------

`RiemannianSGD(params, lr=<required parameter>, rgrad=<required parameter>, expm=<required parameter>)`
:   Riemannian stochastic gradient descent.
    
    Args:
        rgrad (Function): Function to compute the Riemannian gradient
           from the Euclidean gradient
        retraction (Function): Function to update the retraction
           of the Riemannian gradient

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, lr=None, counts=None, **kwargs)`
    :   Performs a single optimization step.
        
        Arguments:
            lr (float, optional): learning rate for the current update.