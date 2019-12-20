---
id: lorentz
title: Module hype.lorentz
sidebar_label: Module hype.lorentz
---
Functions
---------

    
`acosh(...)`
:   

Classes
-------

`LorentzDot(*args, **kwargs)`
:   Records operation history and defines formulas for differentiating ops.
    
    Every operation performed on :class:`Tensor` s creates a new function
    object, that performs the computation, and records that it happened.
    The history is retained in the form of a DAG of functions, with edges
    denoting data dependencies (``input <- output``). Then, when backward is
    called, the graph is processed in the topological ordering, by calling
    :func:`backward` methods of each :class:`Function` object, and passing
    returned gradients on to next :class:`Function` s.
    
    Normally, the only way users interact with functions is by creating
    subclasses and defining new operations. This is a recommended way of
    extending torch.autograd.
    
    Each function object is meant to be used only once (in the forward pass).
    
    Examples::
    
        >>> class Exp(Function):
        >>>
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch._C._FunctionBase
    * torch.autograd.function._ContextMethodMixin
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx, g)`
    :   Defines a formula for differentiating the operation.
        
        This function is to be overridden by all subclasses.
        
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs did :func:`forward` return, and it should return as many
        tensors, as there were inputs to :func:`forward`. Each argument is the
        gradient w.r.t the given output, and each returned value should be the
        gradient w.r.t. the corresponding input.
        
        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computated w.r.t. the
        output.

    `forward(ctx, u, v)`
    :   Performs the operation.
        
        This function is to be overridden by all subclasses.
        
        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
        
        The context can be used to store tensors that can be then retrieved
        during the backward pass.

`LorentzManifold(eps=1e-12, norm_clip=1, max_norm=1000000.0, debug=False, **kwargs)`
:   

    ### Ancestors (in MRO)

    * hype.manifold.Manifold

    ### Static methods

    `dim(dim)`
    :

    `ldot(u, v, keepdim=False)`
    :   Lorentzian Scalar Product

    ### Instance variables

    `debug`
    :   Return an attribute of instance, which is of type owner.

    `eps`
    :   Return an attribute of instance, which is of type owner.

    `max_norm`
    :   Return an attribute of instance, which is of type owner.

    `norm_clip`
    :   Return an attribute of instance, which is of type owner.

    ### Methods

    `expm(self, p, d_p, lr=None, out=None, normalize=False)`
    :   Exponential map for hyperboloid

    `init_weights(self, w, irange=1e-05)`
    :

    `logm(self, x, y)`
    :   Logarithmic map on the Lorenz Manifold

    `normalize(self, w)`
    :   Normalize vector such that it is located on the hyperboloid

    `normalize_tan(self, x_all, v_all)`
    :

    `pnorm(self, u)`
    :

    `ptransp(self, x, y, v, ix=None, out=None)`
    :   Parallel transport for hyperboloid

    `rgrad(self, p, d_p)`
    :   Riemannian gradient for hyperboloid

    `to_poincare_ball(self, u)`
    :