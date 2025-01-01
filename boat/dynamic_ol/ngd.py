import torch.autograd

from .dynamical_system import DynamicalSystem

from torch.nn import Module
from higher.patch import _MonkeyPatchBase
from higher.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from ..utils.op_utils import stop_grads


class NGD(DynamicalSystem):
    """
    Implements the optimization procedure of the Naive Gradient Descent (NGD) [1].

    Parameters
    ----------
    :param ll_objective: The lower-level objective function of the BLO problem.
    :type ll_objective: Callable
    :param ul_objective: The upper-level objective function of the BLO problem.
    :type ul_objective: Callable
    :param ll_model: The lower-level model of the BLO problem.
    :type ll_model: torch.nn.Module
    :param ul_model: The upper-level model of the BLO problem.
    :type ul_model: torch.nn.Module
    :param lower_loop: The number of iterations for lower-level optimization.
    :type lower_loop: int
    :param solver_config: A dictionary containing configurations for the solver. Expected keys include:
        - "lower_level_opt" (torch.optim.Optimizer): The optimizer for the lower-level model.
        - "hyper_op" (List[str]): A list of hyper-gradient operations to apply, such as "PTT" or "FOA".
        - "RGT" (Dict): Configuration for Truncated Gradient Iteration (RGT):
            - "truncate_iter" (int): The number of iterations to truncate the gradient computation.
    :type solver_config: Dict[str, Any]

    Attributes
    ----------
    :attribute truncate_max_loss_iter: Whether to truncate based on a maximum loss iteration (enabled if "PTT" is in `hyper_op`).
    :type truncate_max_loss_iter: bool
    :attribute truncate_iters: Number of iterations for gradient truncation, derived from `solver_config["RGT"]["truncate_iter"]`.
    :type truncate_iters: int
    :attribute ll_opt: The optimizer used for the lower-level model.
    :type ll_opt: torch.optim.Optimizer
    :attribute foa: Whether First-Order Approximation (FOA) is applied, based on `hyper_op` configuration.
    :type foa: bool

    References
    ----------
    [1] L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel
        programming for hyperparameter optimization and meta-learning", in ICML, 2018.
    """


    def __init__(
        self,
        ll_objective: Callable,
        ul_objective: Callable,
        ll_model: Module,
        ul_model: Module,
        lower_loop: int,
        solver_config: Dict[str, Any],
    ):

        super(NGD, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]
        self.truncate_iters = solver_config["RGT"]["truncate_iter"]
        self.ll_opt = solver_config["lower_level_opt"]
        self.foa = "FOA" in solver_config["hyper_op"]

    def optimize(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        auxiliary_opt: DifferentiableOptimizer,
        current_iter: int,
        next_operation: str = None,
        **kwargs
    ):
        """
        Execute the lower-level optimization procedure using data, models, and patched optimizers.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization.
            Typically includes training data, targets, and other information required to compute the lower-level (LL) objective.

        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization.
            Typically includes validation data, targets, and other information required to compute the upper-level (UL) objective.

        auxiliary_model : _MonkeyPatchBase
            A patched lower-level model wrapped by the `higher` library.
            Used for differentiable optimization in the lower-level procedure.

        auxiliary_opt : DifferentiableOptimizer
            A patched optimizer for the lower-level model, wrapped by the `higher` library.
            Enables differentiable optimization.

        current_iter : int
            The current iteration number of the optimization process.

        Returns
        -------
        None
        """
        
        assert next_operation is None, "NGD does not support next_operation"
        if "gda_loss" in kwargs:
            gda_loss = kwargs["gda_loss"]
            alpha = kwargs["alpha"]
            alpha_decay = kwargs["alpha_decay"]
        else:
            gda_loss = None
        if self.truncate_iters > 0:
            ll_backup = [
                x.data.clone().detach().requires_grad_()
                for x in self.ll_model.parameters()
            ]
            for lower_iter in range(self.truncate_iters):
                if gda_loss is not None:
                    ll_feed_dict["alpha"] = alpha
                    loss_f = gda_loss(
                        ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                    )
                    alpha = alpha * alpha_decay
                else:
                    loss_f = self.ll_objective(
                        ll_feed_dict, self.ul_model, auxiliary_model
                    )

                loss_f.backward()
                self.ll_opt.step()
                self.ll_opt.zero_grad()
            for x, y in zip(self.ll_model.parameters(), auxiliary_model.parameters()):
                y.data = x.data.clone().detach().requires_grad_()
            for x, y in zip(ll_backup, self.ll_model.parameters()):
                y.data = x.data.clone().detach().requires_grad_()

        # truncate with PTT method
        if self.truncate_max_loss_iter:
            ul_loss_list = []
            for lower_iter in range(self.lower_loop):

                if gda_loss is not None:
                    ll_feed_dict["alpha"] = alpha
                    loss_f = gda_loss(
                        ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                    )
                    alpha = alpha * alpha_decay
                else:
                    loss_f = self.ll_objective(
                        ll_feed_dict, self.ul_model, auxiliary_model
                    )
                auxiliary_opt.step(loss_f)

                upper_loss = self.ul_objective(
                    ul_feed_dict, self.ul_model, auxiliary_model
                )
                ul_loss_list.append(upper_loss.item())
            ll_step_with_max_ul_loss = ul_loss_list.index(max(ul_loss_list))
            return ll_step_with_max_ul_loss + 1

        for lower_iter in range(self.lower_loop - self.truncate_iters):
            if gda_loss is not None:
                ll_feed_dict["alpha"] = alpha
                loss_f = gda_loss(
                    ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                )
                alpha = alpha * alpha_decay
            else:
                loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
            auxiliary_opt.step(loss_f, grad_callback=stop_grads if self.foa else None)
        return self.lower_loop - self.truncate_iters
