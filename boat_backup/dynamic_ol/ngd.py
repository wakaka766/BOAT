import mindspore
from mindspore import nn, Tensor, Parameter, value_and_grad
from mindspore.common.initializer import initializer
from typing import Callable, Dict, Any
from .dynamical_system import DynamicalSystem
from ..utils.op_utils import stop_grads


class NGD(DynamicalSystem):
    """
    Implements the lower-level optimization procedure of the Naive Gradient Descent (NGD) _`[1]`.

    Parameters
    ----------
        :param ll_objective: The lower-level objective of the BLO problem.
        :type ll_objective: callable
        :param ul_objective: The upper-level objective of the BLO problem.
        :type ul_objective: callable
        :param ll_model: The lower-level model of the BLO problem.
        :type ll_model: mindspore.nn.Cell
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: mindspore.nn.Cell
        :param lower_loop: Number of iterations for lower-level optimization.
        :type lower_loop: int
        :param solver_config: Dictionary containing solver configurations.
        :type solver_config: dict

    References
    ----------
    _`[1]` L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel
     programming for hyperparameter optimization and meta-learning", in ICML, 2018.
    """

    def __init__(
        self,
        ll_objective: Callable,
        ul_objective: Callable,
        ll_model: nn.Cell,
        ul_model: nn.Cell,
        lower_loop: int,
        solver_config: Dict[str, Any]
    ):
        super(NGD, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]
        self.truncate_iters = solver_config['RGT']["truncate_iter"]
        self.ul_objective = ul_objective
        self.ll_opt = solver_config['ll_opt']
        self.foa = 'FOA' in solver_config['hyper_op']

    def optimize(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: nn.Cell,
        auxiliary_opt: nn.Optimizer,
        current_iter: int
    ):
        """
        Execute the lower-level optimization procedure with the data from feed_dict and patched models.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param auxiliary_model: A patched lower model wrapped by MindSpore.
            It serves as the lower-level model for optimization.
        :type auxiliary_model: mindspore.nn.Cell

        :param auxiliary_opt: A patched optimizer for the lower-level model.
            This optimizer allows for differentiable optimization.
        :type auxiliary_opt: mindspore.nn.Optimizer

        :param current_iter: The current iteration number of the optimization process.
        :type current_iter: int

        :returns: None
        """

        def compute_grads(loss_fn, model):
            """Compute gradients for model parameters using MindSpore."""
            grad_fn = value_and_grad(loss_fn, grad_position=0, weights=model.trainable_params(), has_aux=False)
            loss, grads = grad_fn()
            return loss, grads

        # Truncation with RGT method
        if self.truncate_iters > 0:
            ll_backup = [param.clone().asnumpy() for param in self.ll_model.trainable_params()]
            for _ in range(self.truncate_iters):
                lower_loss, grads = compute_grads(lambda: self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model), self.ll_model)
                auxiliary_opt(grads)
            for param, aux_param in zip(self.ll_model.trainable_params(), auxiliary_model.trainable_params()):
                aux_param.set_data(Tensor(param.clone()))
            for backup, param in zip(ll_backup, self.ll_model.trainable_params()):
                param.set_data(Tensor(backup))

        # Truncate with PTT method
        if self.truncate_max_loss_iter:
            ul_loss_list = []
            for _ in range(self.lower_loop):
                lower_loss, grads = compute_grads(lambda: self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model), auxiliary_model)
                auxiliary_opt(grads)
                upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
                ul_loss_list.append(upper_loss.asnumpy())
            ll_step_with_max_ul_loss = ul_loss_list.index(max(ul_loss_list))
            return ll_step_with_max_ul_loss + 1

        # Standard lower-loop optimization
        for _ in range(self.lower_loop - self.truncate_iters):
            lower_loss, grads = compute_grads(lambda: self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model), auxiliary_model)
            auxiliary_opt(grads, grad_callback=stop_grads if self.foa else None)

        return self.lower_loop - self.truncate_iters
