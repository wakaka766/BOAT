import torch
from .hyper_gradient import HyperGradient
from torch.nn import Module
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase
from boat.utils.op_utils import update_tensor_grads


class FD(HyperGradient):
    """
    Calculation of the hyper gradient of the upper-level variables with Finite Differentiation (FD) _`[1]`.

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
    _`[1]` H. Liu, K. Simonyan, Y. Yang, "DARTS: Differentiable Architecture Search",
     in ICLR, 2019.
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
        super(FD, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )
        self.ll_lr = solver_config["lower_level_opt"].defaults["lr"]
        self.dynamic_initialization = "DI" in solver_config["dynamic_op"]
        self._r = solver_config["FD"]["r"]
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.gda_loss = solver_config.get("gda_loss", None)

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
        lower_model_params = kwargs.get(
            "lower_model_params", list(auxiliary_model.parameters())
        )
        loss = self.ul_objective(
            ul_feed_dict, self.ul_model, auxiliary_model, params=lower_model_params
        )
        grad_x = torch.autograd.grad(loss, list(self.ul_var), retain_graph=True)
        grad_y = torch.autograd.grad(
            loss,
            list(auxiliary_model.parameters()),
            retain_graph=self.dynamic_initialization,
        )

        dalpha = [v.data for v in grad_x]
        vector = [v.data for v in grad_y]
        implicit_grads = self._hessian_vector_product(
            vector, ll_feed_dict, ul_feed_dict
        )

        for g, ig in zip(dalpha, implicit_grads):
            g.sub_(ig.data, alpha=self.ll_lr)

        if self.dynamic_initialization:
            grads_lower = torch.autograd.grad(
                loss, list(auxiliary_model.parameters(time=0))
            )
            update_tensor_grads(self.ll_var, grads_lower)

        update_tensor_grads(self.ul_var, dalpha)

        return {"upper_loss": loss, "hyper_gradient_finished": True}

    def _hessian_vector_product(self, vector, ll_feed_dict, ul_feed_dict):
        """
        Built-in calculation function. Compute the first order approximation of
        the second-order derivative of upper variables.

        Parameters
        ----------
           train_data: Tensor
                The training data used for upper level problem optimization.

            train_target: Tensor
                The labels of the samples in the train data.

        Returns
        -------
        Tensor
           Returns the calculated first order approximation grads.
        """
        eta = self._r / torch.cat([x.view(-1) for x in vector]).norm()
        for p, v in zip(self.ll_model.parameters(), vector):
            p.data.add_(v, alpha=eta)  # w+
        if self.gda_loss is not None:
            ll_feed_dict["alpha"] = self.alpha
            loss = self.gda_loss(
                ll_feed_dict, ul_feed_dict, self.ul_model, self.ll_model
            )
        else:
            loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        grads_p = torch.autograd.grad(loss, list(self.ul_var))

        for p, v in zip(self.ll_model.parameters(), vector):
            p.data.sub_(v, alpha=2 * eta)  # w-
        if self.gda_loss is not None:
            ll_feed_dict["alpha"] = self.alpha
            loss = self.gda_loss(
                ll_feed_dict, ul_feed_dict, self.ul_model, self.ll_model
            )
        else:
            loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        grads_n = torch.autograd.grad(loss, list(self.ul_var))

        for p, v in zip(self.ll_model.parameters(), vector):
            p.data.add_(v, alpha=eta)  # w

        return [(x - y).div_(2 * eta) for x, y in zip(grads_p, grads_n)]
