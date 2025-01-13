from typing import List, Callable
from mindspore import ops


def l2_reg(params):
    """
    Compute the L2 regularization loss.

    Parameters
    ----------
    params : list
        List of model parameters (trainable).

    Returns
    -------
    mindspore.Tensor
        The computed L2 regularization loss.
    """
    loss = 0.0
    for param in params:
        loss += ops.ReduceSum()(ops.Pow()(param, 2))
    return loss


def require_model_grad(model=None):
    """
    Ensure all parameters of a MindSpore model require gradients.

    Parameters
    ----------
    model : mindspore.nn.Cell, optional
        MindSpore model instance. Must not be None.

    Raises
    ------
    AssertionError
        If the model is None.
    """
    assert model is not None, "The module is not defined!"
    for param in model.trainable_params():
        param.requires_grad = True


def update_grads(grads, model):
    """
    Update gradients for a model's parameters.

    Parameters
    ----------
    grads : list
        List of gradients to apply.
    model : mindspore.nn.Cell
        The model whose gradients will be updated.
    """
    for grad, param in zip(grads, model.trainable_params()):
        if param.grad is None:
            param.set_grad(grad)
        else:
            param.grad += grad


def update_tensor_grads(hparams, grads):
    """
    Update gradients for hyperparameters.

    Parameters
    ----------
    hparams : list of mindspore.Tensor
        Hyperparameters to update.
    grads : list of mindspore.Tensor
        Gradients to apply to the hyperparameters.
    """
    for param, grad in zip(hparams, grads):
        if param.grad is None:
            param.set_grad(grad)
        else:
            param.grad += grad


def stop_grads(grads):
    """
    Detach and stop gradient computation for a list of gradients.

    Parameters
    ----------
    grads : list of mindspore.Tensor
        Gradients to process.

    Returns
    -------
    list of mindspore.Tensor
        Detached gradients with requires_grad set to False.
    """
    return [
        (grad.detach().requires_grad_(False) if grad is not None else grad)
        for grad in grads
    ]


def average_grad(model, batch_size):
    """
    Average the gradients of all model parameters by the batch size.

    Parameters
    ----------
    model : mindspore.nn.Cell
        The model whose gradients need to be averaged.
    batch_size : int
        The batch size to divide gradients by.
    """
    for param in model.trainable_params():
        if param.grad is not None:
            param.grad /= batch_size


def stop_model_grad(model=None):
    """
    Stop gradient computation for all parameters in a model.

    Parameters
    ----------
    model : mindspore.nn.Cell, optional
        The model to stop gradients for. Must not be None.

    Raises
    ------
    AssertionError
        If the model is None.
    """
    assert model is not None, "The module is not defined!"
    for param in model.trainable_params():
        param.requires_grad = False


def copy_parameter_from_list(model, param_list):
    """
    Copy parameters from a list to a model's trainable parameters.

    Parameters
    ----------
    model : mindspore.nn.Cell
        The model whose parameters need to be updated.
    param_list : list of mindspore.Tensor
        The list of parameters to copy.
    """
    for param, new_param in zip(model.trainable_params(), param_list):
        param.set_data(new_param)
