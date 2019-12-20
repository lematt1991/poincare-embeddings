---
id: euclidean
title: Module hype.euclidean
sidebar_label: Module hype.euclidean
---
Classes
-------

`EuclideanManifold(max_norm=1, **kwargs)`
:   

    ### Ancestors (in MRO)

    * hype.manifold.Manifold

    ### Descendants

    * hype.euclidean.TranseManifold
    * hype.poincare.PoincareManifold

    ### Instance variables

    `max_norm`
    :   Return an attribute of instance, which is of type owner.

    ### Methods

    `normalize(self, u)`
    :

    `pnorm(self, u, dim=-1)`
    :

    `rgrad(self, p, d_p)`
    :

`TranseManifold(dim, *args, **kwargs)`
:   

    ### Ancestors (in MRO)

    * hype.euclidean.EuclideanManifold
    * hype.manifold.Manifold