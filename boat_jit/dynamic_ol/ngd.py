from jittor import Module
from ..higher_jit.patch import _MonkeyPatchBase
from ..higher_jit.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from ..utils.op_utils import stop_grads

from boat_jit.operation_registry import register_class
from boat_jit.dynamic_ol.dynamical_system import DynamicalSystem


@register_class
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
        :type ll_model: torch.nn.Module
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: torch.nn.Module
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
        ll_model: Module,
        ul_model: Module,
        lower_loop: int,
        solver_config: Dict[str, Any],
    ):

        super(NGD, self).__init__(ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config)
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
        Execute the lower-level optimization procedure with the data from feed_dict and patched models.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param auxiliary_model: A patched lower model wrapped by the `higher` library.
            It serves as the lower-level model for optimization.
        :type auxiliary_model: _MonkeyPatchBase

        :param auxiliary_opt: A patched optimizer for the lower-level model,
            wrapped by the `higher` library. This optimizer allows for differentiable optimization.
        :type auxiliary_opt: DifferentiableOptimizer

        :param current_iter: The current iteration number of the optimization process.
        :type current_iter: int

        :param next_operation: The next operation to be executed in the optimization process.
        :type next_operation: str

        :param kwargs: Additional arguments for the optimization process.
        :type kwargs: dict

        :returns: None
        """

        assert next_operation is None, "NGD does not support next_operation"
        if "gda_loss" in kwargs:
            gda_loss = kwargs["gda_loss"]
            alpha = kwargs["alpha"]
            alpha_decay = kwargs["alpha_decay"]
        else:
            gda_loss = None

        # truncate with RGT operation
        if self.truncate_iters > 0:
            ll_backup = [x.clone().stop_grad() for x in self.ll_model.parameters()]
            for _ in range(self.truncate_iters):
                if gda_loss is not None:
                    ll_feed_dict["alpha"] = alpha
                    loss_f = gda_loss(
                        ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                    )
                    alpha = alpha * alpha_decay
                else:
                    loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
                self.ll_opt.step(loss_f)
            for x, y in zip(self.ll_model.parameters(), auxiliary_model.parameters()):
                y.update(x.clone())
            for x, y in zip(ll_backup, self.ll_model.parameters()):
                y.update(x.clone())

        # truncate with PTT method
        if self.truncate_max_loss_iter:
            ul_loss_list = []
            for _ in range(self.lower_loop):
                if gda_loss is not None:
                    ll_feed_dict["alpha"] = alpha
                    loss_f = gda_loss(
                        ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                    )
                    alpha = alpha * alpha_decay
                else:
                    loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
                auxiliary_opt.step(loss_f)
                upper_loss = self.ul_objective(
                    ul_feed_dict, self.ul_model, auxiliary_model
                )
                ul_loss_list.append(upper_loss.item())
            ll_step_with_max_ul_loss = ul_loss_list.index(max(ul_loss_list))
            return ll_step_with_max_ul_loss + 1

        for _ in range(self.lower_loop - self.truncate_iters):
            if gda_loss is not None:
                ll_feed_dict["alpha"] = alpha
                loss_f = gda_loss(
                    ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                )
                alpha = alpha * alpha_decay
            else:
                loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
            auxiliary_opt.step(loss_f)
            auxiliary_opt.step(loss_f, grad_callback=stop_grads if self.foa else None)
        return -1
