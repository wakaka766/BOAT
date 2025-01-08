import torch
from torch import Tensor
from typing import List, Callable, Dict


class ResultStore:
    """
    A simple class to store and manage intermediate results of hyper-gradient computation.
    """

    def __init__(self):
        self.results = []

    def add(self, name: str, result: Dict):
        """
        Add a result to the store.

        :param name: The name of the result (e.g., 'gradient_operator_results_0').
        :type name: str
        :param result: The result dictionary to store.
        :type result: Dict
        """
        self.results.append({name: result})

    def clear(self):
        """Clear all stored results."""
        self.results = []

    def get_results(self) -> List[Dict]:
        """Retrieve all stored results."""
        return self.results


class DynamicalSystemRules:
    """
    A class to store and manage gradient operator rules.
    """

    # Default static gradient operator order
    _gradient_order = [
        ["GDA", "DI"],
        ["DM", "NGD"],
    ]

    @staticmethod
    def get_gradient_order() -> List[List[str]]:
        """
        Get the current gradient operator order.

        Returns
        -------
        List[List[str]]
            The current gradient operator order.
        """
        return DynamicalSystemRules._gradient_order

    @staticmethod
    def set_gradient_order(new_order: List[List[str]]):
        """
        Set a new gradient operator order.

        Parameters
        ----------
        new_order : List[List[str]]
            The new gradient operator order to set.

        Raises
        ------
        ValueError
            If the new order is invalid.
        """
        if not isinstance(new_order, list) or not all(
            isinstance(group, list) for group in new_order
        ):
            raise ValueError("Gradient order must be a list of lists.")
        DynamicalSystemRules._gradient_order = new_order


class HyperGradientRules:
    """
    A class to store and manage gradient operator rules.
    """

    # Default static gradient operator order
    _gradient_order = [
        ["PTT", "FOA", "RGT"],
        ["IAD", "RAD", "FD", "IGA"],
        ["CG", "NS"],
    ]

    @staticmethod
    def get_gradient_order() -> List[List[str]]:
        """
        Get the current gradient operator order.

        Returns
        -------
        List[List[str]]
            The current gradient operator order.
        """
        return HyperGradientRules._gradient_order

    @staticmethod
    def set_gradient_order(new_order: List[List[str]]):
        """
        Set a new gradient operator order.

        Parameters
        ----------
        new_order : List[List[str]]
            The new gradient operator order to set.

        Raises
        ------
        ValueError
            If the new order is invalid.
        """
        if not isinstance(new_order, list) or not all(
            isinstance(group, list) for group in new_order
        ):
            raise ValueError("Gradient order must be a list of lists.")
        HyperGradientRules._gradient_order = new_order


def l2_reg(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def grad_unused_zero(
    output, inputs, grad_outputs=None, retain_graph=False, create_graph=False
):
    grads = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )

    def grad_or_zeros(grad, var):
        return (
            torch.zeros_like(var) if grad is None or (torch.isnan(grad).any()) else grad
        )

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


def require_model_grad(model=None):
    assert model is not None, "The module is not defined!"
    for param in model.parameters():
        param.requires_grad_(True)


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
    return [
        (grad.detach().requires_grad_(False) if grad is not None else grad)
        for grad in grads
    ]


def average_grad(model, batch_size):
    for param in model.parameters():
        param.grad.data = param.grad.data / batch_size


def stop_model_grad(model=None):
    assert model is not None, "The module is not defined!"
    for param in model.parameters():
        param.requires_grad_(False)


def copy_parameter_from_list(y, z):
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    return y


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(
        outer_loss, hparams, retain_graph=retain_graph
    )

    return grad_outer_w, grad_outer_hparams


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


def neumann(
    params: List[Tensor],
    hparams: List[Tensor],
    upper_loss,
    lower_loss,
    k: int,
    fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
    tol=1e-10,
) -> List[Tensor]:

    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)

    w_mapped = fp_map(params, lower_loss)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for i in range(k):
        gs_prev_vec = gs_vec
        vs = torch.autograd.grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(torch.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = torch.autograd.grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    return grads


def conjugate_gradient(
    params: List[Tensor],
    hparams: List[Tensor],
    upper_loss,
    lower_loss,
    K: int,
    fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
    tol=1e-10,
) -> List[Tensor]:
    import time
    starttime = time.time()
    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)
    print("step 2 time:", time.time() - starttime)
    starttime = time.time()

    w_mapped = fp_map(params, lower_loss)
    print("step 3 time:", time.time() - starttime)
    def dfp_map_dw(xs):
        Jfp_mapTv = torch.autograd.grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
        return [v - j for v, j in zip(xs, Jfp_mapTv)]
    starttime = time.time()
    vs = cg_step(
        dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol
    )
    print("step 4 cg time:", time.time() - starttime)
    starttime = time.time()
    grads = torch.autograd.grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    print("step 5 grad time:", time.time() - starttime)

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
