from typing import List, Callable
from mindspore import ops

def l2_reg(params):
    """
    Compute the L2 regularization loss.

    Args:
        params (list): List of model parameters (trainable).

    Returns:
        mindspore.Tensor: The computed L2 regularization loss.
    """
    loss = 0.0
    for param in params:
        loss += ops.ReduceSum()(ops.Pow()(param, 2))  # Equivalent to torch.norm(param, 2) ** 2
    return loss


def require_model_grad(model=None):
    """
    Ensure all parameters of a MindSpore model require gradients.

    :param model: MindSpore model instance.
    :type model: mindspore.nn.Cell
    """
    assert model is not None, 'The module is not defined!'
    for param in model.trainable_params():  # 使用 trainable_params 替代 parameters
        param.requires_grad = True  # MindSpore 中通过直接设置属性修改

def update_grads(grads, model):
    for p, x in zip(grads, model.parameters()):
        if x.grad is None:
            x.grad = p
        else:
            x.grad += p


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = g
        else:
            l.grad += g


def stop_grads(grads):
    return [(grad.detach().requires_grad_(False) if grad is not None else grad) for grad in grads]


def average_grad(model, batch_size):
    for param in model.parameters():
        param.grad.data = param.grad.data / batch_size


def stop_model_grad(model=None):
    assert model is not None, 'The module is not defined!'
    for param in model.parameters():
        param.requires_grad_(False)


def copy_parameter_from_list(model, param_list):
    """
    Copy parameters from a list to a model's trainable parameters.

    Args:
        model (mindspore.nn.Cell): The model whose parameters need to be updated.
        param_list (list of mindspore.Tensor): The list of parameters to copy.
    """
    for param, new_param in zip(model.trainable_params(), param_list):
        param.set_data(new_param)


