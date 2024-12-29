from .dynamical_system import DynamicalSystem
from torch.nn import Module
from higher.patch import _MonkeyPatchBase
from higher.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from boat.utils.op_utils import stop_grads


class DI_GDA_NGD(DynamicalSystem):
    """
    Implements the lower-level optimization procedure of the Naive Gradient Descent (NGD) _`[1]`, Gradient Descent
    Aggregation (GDA) _`[2]` and Dynamic Initialization (DI) _`[3]`.

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

    _`[2]` R. Liu, P. Mu, X. Yuan, S. Zeng, and J. Zhang, "A generic first-order algorithmic
     framework for bi-level programming beyond lower-level singleton", in ICML, 2020.

    _`[3]` R. Liu, Y. Liu, S. Zeng, and J. Zhang, "Towards Gradient-based Bilevel
     Optimization with Non-convex Followers and Beyond", in NeurIPS, 2021.
    """

    def __init__(
        self,
        ll_objective: Callable,
        lower_loop: int,
        ul_model: Module,
        ul_objective: Callable,
        ll_model: Module,
        solver_config: Dict[str, Any],
    ):

        super(DI_GDA_NGD, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]
        self.ul_objective = ul_objective
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.truncate_iters = solver_config["RGT"]["truncate_iter"]
        self.ll_opt = solver_config["ll_opt"]
        self.gda_loss = solver_config["gda_loss"]
        self.foa = "FOA" in solver_config["hyper_op"]

    def optimize(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        auxiliary_opt: DifferentiableOptimizer,
        current_iter: int,
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

        :returns: None
        """

        alpha = self.alpha

        # truncate with T-RAD method
        if self.truncate_iters > 0:
            ll_backup = [
                x.data.clone().detach().requires_grad_()
                for x in self.ll_model.parameters()
            ]
            for lower_iter in range(self.truncate_iters):
                # lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
                assert (self.alpha > 0) and (
                    self.alpha < 1
                ), "Set the coefficient alpha properly in (0,1)."
                assert (
                    self.gda_loss is not None
                ), "Define the gda_loss properly in loss_func.py."
                ll_feed_dict["alpha"] = alpha
                loss_f = self.gda_loss(
                    ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                )
                alpha = alpha * self.alpha_decay
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
                assert (self.alpha > 0) and (
                    self.alpha < 1
                ), "Set the coefficient alpha properly in (0,1)."
                assert (
                    self.gda_loss is not None
                ), "Define the gda_loss properly in loss_func.py."
                ll_feed_dict["alpha"] = alpha
                loss_f = self.gda_loss(
                    ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                )
                auxiliary_opt.step(loss_f)
                alpha = alpha * self.alpha_decay
                upper_loss = self.ul_objective(
                    ul_feed_dict, self.ul_model, auxiliary_model
                )
                ul_loss_list.append(upper_loss.item())
            ll_step_with_max_ul_loss = ul_loss_list.index(max(ul_loss_list))
            return ll_step_with_max_ul_loss + 1

        for lower_iter in range(self.lower_loop - self.truncate_iters):
            assert (self.alpha > 0) and (
                self.alpha < 1
            ), "Set the coefficient alpha properly in (0,1)."
            assert (
                self.gda_loss is not None
            ), "Define the gda_loss properly in loss_func.py."
            ll_feed_dict["alpha"] = alpha
            loss_f = self.gda_loss(
                ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
            )
            auxiliary_opt.step(loss_f, grad_callback=stop_grads if self.foa else None)
            alpha = alpha * self.alpha_decay

        return self.lower_loop - self.truncate_iters
