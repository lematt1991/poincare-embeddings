---
id: sn
title: Module hype.sn
sidebar_label: Module hype.sn
---
Functions
---------

    
`initialize(manifold, opt, idx, objects, weights, sparse=True)`
:   

    
`randint(...)`
:   randint(low, high=None, size=None, dtype='l')
    
    Return random integers from `low` (inclusive) to `high` (exclusive).
    
    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. All dtypes are determined by their
        name, i.e., 'int64', 'int', etc, so byteorder is not available
        and a specific precision may have different C types depending
        on the platform. The default value is 'np.int'.
    
        .. versionadded:: 1.11.0
    
    Returns
    -------
    out : int or ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.
    
    See Also
    --------
    random.random_integers : similar to `randint`, only for the closed
        interval [`low`, `high`], and 1 is the lowest value if `high` is
        omitted. In particular, this other one is the one to use to generate
        uniformly distributed discrete non-integers.
    
    Examples
    --------
    >>> np.random.randint(2, size=10)
    array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
    >>> np.random.randint(1, size=10)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    Generate a 2 x 4 array of ints between 0 and 4, inclusive:
    
    >>> np.random.randint(5, size=(2, 4))
    array([[4, 0, 2, 1],
           [3, 2, 2, 0]])

Classes
-------

`Dataset(idx, objects, weights, nnegs, unigram_size=100000000.0)`
:   An abstract class representing a Dataset.
    
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.

    ### Ancestors (in MRO)

    * hype.graph.Dataset
    * torch.utils.data.dataset.Dataset

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

    * hype.graph.Embedding
    * torch.nn.modules.module.Module

    ### Methods

    `loss(self, preds, targets, weight=None, size_average=True)`
    :