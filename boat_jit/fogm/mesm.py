from boat_jit.utils.op_utils import (
    grad_unused_zero,
    update_tensor_grads,
    copy_parameter_from_list,
    manual_update,
)
import numpy
from jittor import Module
import copy
from typing import Dict, Any, Callable, List
import jittor as jit

from boat_jit.operation_registry import register_class
from boat_jit.dynamic_ol.dynamical_system import DynamicalSystem


@register_class
class MESM(DynamicalSystem):
    """
    Implements the optimization procedure of Moreau Envelop based Single-loop Method (MESM).

    Parameters
    ----------
    :param ll_objective: The lower-level objective of the BLO problem.
    :type ll_objective: callable
    :param ul_objective: The upper-level objective of the BLO problem.
    :type ul_objective: callable
    :param ll_model: The lower-level model of the BLO problem.
    :type ll_model: torch.nn.Module
    :param ul_model: The upper-level model of the BLO problem.
    :type ul_model: torch.nn.Module
    :param ll_var: The list of lower-level variables of the BLO problem.
    :type ll_var: List
    :param ul_var: The list of upper-level variables of the BLO problem.
    :type ul_var: List
    :param lower_loop: Number of iterations for lower-level optimization.
    :type lower_loop: int
    :param solver_config: Dictionary containing solver configurations.
    :type solver_config: dict


    References
    ----------
    _`[1]` Liu R, Liu Z, Yao W, et al. Moreau Envelope for Nonconvex Bi-Level Optimization: A Single-loop and
    Hessian-free Solution Strategy[J]. ICML, 2024.
    """

    def __init__(
        self,
        ll_objective: Callable,
        lower_loop: int,
        ul_model: Module,
        ul_objective: Callable,
        ll_model: Module,
        ll_var: List,
        ul_var: List,
        solver_config: Dict[str, Any],
    ):
        super(MESM, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_loop = lower_loop
        self.eta = solver_config["MESM"]["eta"]
        self.gamma_1 = solver_config["MESM"]["gamma_1"]
        self.c0 = solver_config["MESM"]["c0"]
        self.y_hat = copy.deepcopy(self.ll_model)
        self.y_hat_opt = jit.optim.SGD(
            self.y_hat.parameters(), lr=solver_config["MESM"]["y_hat_lr"], momentum=0.9
        )

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        """
        Execute the optimization procedure with the data from feed_dict.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param current_iter: The current iteration number of the optimization process.
        :type current_iter: int

        :returns: None
        """

        if current_iter == 0:
            ck = 0.2
        else:
            ck = numpy.power(current_iter + 1, 0.25) * self.c0

        theta_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.y_hat)

        grad_theta_parmaters = grad_unused_zero(
            theta_loss, list(self.y_hat.parameters())
        )

        errs = []
        for a, b in zip(
            list(self.y_hat.parameters()), list(self.ll_model.parameters())
        ):
            diff = a - b
            errs.append(diff)
        vs_param = []
        for v0, gt, err in zip(
            list(self.y_hat.parameters()), grad_theta_parmaters, errs
        ):
            vs_param.append(v0 - self.eta * (gt + self.gamma_1 * err))  # upate \theta

        copy_parameter_from_list(self.y_hat, vs_param)

        reg = 0
        for param1, param2 in zip(list(self.ll_model.parameters()), vs_param):
            diff = param1 - param2
            reg += (diff**2).sum()  # Jittor-compatible L2 norm calculation
        lower_loss = (
            (1 / ck) * self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
            + self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            - 0.5 * self.gamma_1 * reg
        )

        grad_y_parmaters = grad_unused_zero(
            lower_loss, list(self.ll_model.parameters())
        )

        update_tensor_grads(self.ll_var, grad_y_parmaters)

        manual_update(self.ll_opt, self.ll_var)
        upper_loss = (
            (1 / ck) * self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
            + self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            - self.ll_objective(ll_feed_dict, self.ul_model, self.y_hat)
        )
        grad_x_parmaters = grad_unused_zero(
            upper_loss, list(self.ul_model.parameters())
        )
        update_tensor_grads(self.ul_var, grad_x_parmaters)

        return upper_loss
