---
id: manifold
title: Module hype.manifold
sidebar_label: Module hype.manifold
---
Classes
-------

`Manifold(*args, **kwargs)`
:   

    ### Descendants

    * hype.euclidean.EuclideanManifold
    * hype.lorentz.LorentzManifold

    ### Static methods

    `dim(dim)`
    :

    ### Methods

    `distance(self, u, v)`
    :   Distance function

    `expm(self, p, d_p, lr=None, out=None)`
    :   Exponential map

    `init_weights(self, w, scale=0.0001)`
    :

    `logm(self, x, y)`
    :   Logarithmic map

    `normalize(self, u)`
    :

    `ptransp(self, x, y, v, ix=None, out=None)`
    :   Parallel transport