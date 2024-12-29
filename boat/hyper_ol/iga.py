import torch
from .hyper_gradient import HyperGradient
from torch.nn import Module
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase
from boat.utils.op_utils import update_tensor_grads


class IGA(HyperGradient):
    """
    Calculation of the hyper gradient of the upper-level variables with  Implicit Gradient Approximation (IGA) _`[1]`.

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
        :param ll_var: List of variables optimized with the lower-level objective.
        :type ll_var: List
        :param ul_var:  of variables optimized with the upper-level objective.
        :type ul_var: List
        :param solver_config: Dictionary containing solver configurations.
        :type solver_config: dict

    References
    ----------
    _`[1]` Liu R, Gao J, Liu X, et al. Learning with constraint learning: New perspective, solution strategy and
    various applications[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
    """

    def __init__(
        self,
        ll_objective: Callable,
        ul_objective: Callable,
        ll_model: Module,
        ul_model: Module,
        ll_var: List,
        ul_var: List,
        solver_config: Dict,
    ):
        super(IGA, self).__init__(ll_objective, ul_objective, ul_model, ll_model, ll_var, ul_var, solver_config)
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.gda_loss = solver_config["gda_loss"]
        self.dynamic_initialization = "DI" in solver_config["dynamic_op"]

    def compute_gradients(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        max_loss_iter: int = 0,
        hyper_gradient_finished: bool = False,
        next_operation: str = None,
        **kwargs
    ):
        """
        Compute the hyper-gradients of the upper-level variables with the data from feed_dict and patched models.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param auxiliary_model: A patched lower model wrapped by the `higher` library.
            It serves as the lower-level model for optimization.
        :type auxiliary_model: _MonkeyPatchBase

        :param max_loss_iter: The number of iteration used for backpropagation.
        :type max_loss_iter: int

        :param next_operation: The next operator for the calculation of the hypergradient.
        :type next_operation: str

        :param hyper_gradient_finished: A boolean flag indicating whether the hypergradient computation is finished.
        :type  hyper_gradient_finished: bool

        :returns: the current upper-level objective
        """
        assert next_operation is None, "FD does not support next_operation"
        lower_model_params = kwargs.get("lower_model_params", list(auxiliary_model.parameters()))
        if self.gda_loss is not None:
            ll_feed_dict["alpha"] = self.alpha * self.alpha_decay**max_loss_iter
            lower_loss = self.gda_loss(
                ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model, params=lower_model_params
            )
        else:
            lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model, params=lower_model_params)
        dfy = torch.autograd.grad(
            lower_loss, lower_model_params, retain_graph=True
        )

        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        dFy = torch.autograd.grad(
            upper_loss, lower_model_params, retain_graph=True
        )

        # calculate GN loss
        gFyfy = 0
        gfyfy = 0
        for Fy, fy in zip(dFy, dfy):
            gFyfy = gFyfy + torch.sum(Fy * fy)
            gfyfy = gfyfy + torch.sum(fy * fy)
        GN_loss = -gFyfy.detach() / gfyfy.detach() * lower_loss

        if self.dynamic_initialization:
            grads_lower = torch.autograd.grad(
                upper_loss, list(auxiliary_model.parameters(time=0)), retain_graph=True
            )
            update_tensor_grads(self.ll_var, grads_lower)

        grads_upper = torch.autograd.grad(GN_loss + upper_loss, list(self.ul_var))
        update_tensor_grads(self.ul_var, grads_upper)

        return {'upper_loss': upper_loss, 'hyper_gradient_finished': True}
