---
id: graph
title: Module hype.graph
sidebar_label: Module hype.graph
---
Functions
---------

    
`choice(...)`
:   choice(a, size=None, replace=True, p=None)
    
    Generates a random sample from a given 1-D array
    
            .. versionadded:: 1.7.0
    
    Parameters
    -----------
    a : 1-D array-like or int
        If an ndarray, a random sample is generated from its elements.
        If an int, the random sample is generated as if a were np.arange(a)
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    
    Returns
    --------
    samples : single item or ndarray
        The generated random samples
    
    Raises
    -------
    ValueError
        If a is an int and less than zero, if a or p are not 1-dimensional,
        if a is an array-like of size 0, if p is not a vector of
        probabilities, if a and p have different lengths, or if
        replace=False and the sample size is greater than the population
        size
    
    See Also
    ---------
    randint, shuffle, permutation
    
    Examples
    ---------
    Generate a uniform random sample from np.arange(5) of size 3:
    
    >>> np.random.choice(5, 3)
    array([0, 3, 4])
    >>> #This is equivalent to np.random.randint(0,5,3)
    
    Generate a non-uniform random sample from np.arange(5) of size 3:
    
    >>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
    array([3, 3, 0])
    
    Generate a uniform random sample from np.arange(5) of size 3 without
    replacement:
    
    >>> np.random.choice(5, 3, replace=False)
    array([3,1,0])
    >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]
    
    Generate a non-uniform random sample from np.arange(5) of size
    3 without replacement:
    
    >>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
    array([2, 3, 0])
    
    Any of the above can be repeated with an arbitrary array-like
    instead of just integers. For instance:
    
    >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
    >>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
    array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
          dtype='|S11')

    
`eval_reconstruction(adj, lt, distfn, workers=1, progress=False)`
:   Reconstruction evaluation.  For each object, rank its neighbors by distance
    
    Args:
        adj (dict[int, set[int]]): Adjacency list mapping objects to its neighbors
        lt (torch.Tensor[N, dim]): Embedding table with `N` embeddings and `dim`
            dimensionality
        distfn ((torch.Tensor, torch.Tensor) -> torch.Tensor): distance function.
        workers (int): number of workers to use

    
`eval_reconstruction_slow(adj, lt, distfn)`
:   

    
`load_adjacency_matrix(path, format='hdf5', symmetrize=False)`
:   

    
`load_edge_list(path, symmetrize=False)`
:   

    
`reconstruction_worker(adj, lt, distfn, objects, progress=False)`
:   

Classes
-------

`Dataset(idx, objects, weights, nnegs, unigram_size=100000000.0)`
:   An abstract class representing a Dataset.
    
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset

    ### Descendants

    * hype.sn.Dataset

    ### Static methods

    `collate(batch)`
    :

    ### Methods

    `nnegatives(self)`
    :

    `weights(self, inputs, targets)`
    :

`Embedding(size, dim, manifold, sparse=True)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * hype.sn.Embedding

    ### Methods

    `embedding(self)`
    :

    `forward(self, inputs)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `init_weights(self, manifold, scale=0.0001)`
    :

    `optim_params(self, manifold)`
    :