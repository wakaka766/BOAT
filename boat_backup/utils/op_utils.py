import torch
from torch import Tensor
from typing import List, Callable
from torch.autograd import grad as torch_grad
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

def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None or (torch.isnan(grad).any()) else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def list_tensor_matmul(list1, list2):
    out = 0
    for t1, t2 in zip(list1, list2):
        out = out + torch.sum(t1 * t2)
    return out


def list_tensor_norm(list_tensor, p=2):
    norm = 0
    for t in list_tensor:
        norm = norm + torch.norm(t, p)
    return norm


# def require_model_grad(model=None):
#     assert model is not None, 'The module is not defined!'
#     for param in model.parameters():
#         param.requires_grad_(True)

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


# def copy_parameter_from_list(y, z):
#     for p, q in zip(y.parameters(), z):
#         p.data = q.clone().detach().requires_grad_()
#     return y

def copy_parameter_from_list(model, param_list):
    """
    Copy parameters from a list to a model's trainable parameters.

    Args:
        model (mindspore.nn.Cell): The model whose parameters need to be updated.
        param_list (list of mindspore.Tensor): The list of parameters to copy.
    """
    for param, new_param in zip(model.trainable_params(), param_list):
        param.set_data(new_param)



def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


def neumann(params: List[Tensor],
            hparams: List[Tensor],
            upper_loss,
            lower_loss,
            k: int,
            fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
            tol=1e-10) -> List[Tensor]:

    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)

    w_mapped = fp_map(params, lower_loss)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for i in range(k):
        gs_prev_vec = gs_vec
        vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(torch.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = torch_grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    return grads


def conjugate_gradient(params: List[Tensor],
                       hparams: List[Tensor],
                       upper_loss,
                       lower_loss,
                       K: int,
                       fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                       tol=1e-10,
                       stochastic=False) -> List[Tensor]:
    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, lower_loss)

    def dfp_map_dw(xs):
        if stochastic:
            w_mapped_in = fp_map(params, lower_loss)
            Jfp_mapTv = torch_grad(w_mapped_in, params, grad_outputs=xs, retain_graph=False)
        else:
            Jfp_mapTv = torch_grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
        return [v - j for v, j in zip(xs, Jfp_mapTv)]

    vs = cg_step(dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    if stochastic:
        w_mapped = fp_map(params, lower_loss)

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]

    return grads


def cg_step(Ax, b, max_iter=100, epsilon=1.0e-5):

    x_last = [torch.zeros_like(bb) for bb in b]
    r_last = [torch.zeros_like(bb).copy_(bb) for bb in b]
    p_last = [torch.zeros_like(rr).copy_(rr) for rr in r_last]

    for ii in range(max_iter):
        Ap = Ax(p_last)
        Ap_vec = cat_list_to_tensor(Ap)
        p_last_vec = cat_list_to_tensor(p_last)
        r_last_vec = cat_list_to_tensor(r_last)
        rTr = torch.sum(r_last_vec * r_last_vec)
        pAp = torch.sum(p_last_vec * Ap_vec)
        alpha = rTr / pAp

        x = [xx + alpha * pp for xx, pp in zip(x_last, p_last)]
        r = [rr - alpha * pp for rr, pp in zip(r_last, Ap)]
        r_vec = cat_list_to_tensor(r)

        if float(torch.norm(r_vec)) < epsilon:
            break

        beta = torch.sum(r_vec * r_vec) / rTr
        p = [rr + beta * pp for rr, pp in zip(r, p_last)]

        x_last = x
        p_last = p
        r_last = r

    return x_last

