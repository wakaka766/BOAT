from .dynamical_system import DynamicalSystem
import jittor as jit
from jittor import Module
from ..higher_jit.patch import _MonkeyPatchBase
from ..higher_jit.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from ..utils.op_utils import (
    update_tensor_grads,
    grad_unused_zero,
    list_tensor_norm,
    list_tensor_matmul,
    custom_grad,
    manual_update,
)

from boat_jit.operation_registry import register_class
from boat_jit.dynamic_ol.dynamical_system import DynamicalSystem


@register_class
class DM(DynamicalSystem):
    """
    Implements the lower-level optimization procedure of the Dual Multiplier (DM) _`[1]`.

    Parameters
    ----------
        :param ll_objective: The lower-level objective of the BLO problem.
        :type ll_objective: callable
        :param ul_objective: The upper-level objective of the BLO problem.
        :type ul_objective: callable
        :param ll_model: The lower-level model of the BLO problem.
        :type ll_model: Jittor.Module
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: Jittor.Module
        :param lower_loop: Number of iterations for lower-level optimization.
        :type lower_loop: int
        :param solver_config: Dictionary containing solver configurations.
        :type solver_config: dict


    References
    ----------
    _`[1]` Liu R, Liu Y, Yao W, et al. Averaged method of multipliers for bi-level optimization without lower-level
    strong convexity [C]. In ICML, 2023.
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

        super(DM, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.truncate_iters = solver_config["RGT"]["truncate_iter"]
        self.ll_opt = solver_config["lower_level_opt"]
        self.ul_opt = solver_config["upper_level_opt"]
        self.auxiliary_v = solver_config["DM"]["auxiliary_v"]
        self.auxiliary_v_opt = solver_config["DM"]["auxiliary_v_opt"]
        self.auxiliary_v_lr = solver_config["DM"]["auxiliary_v_lr"]
        self.tau = solver_config["DM"]["tau"]
        self.p = solver_config["DM"]["p"]
        self.mu0 = solver_config["DM"]["mu0"]
        self.eta = solver_config["DM"]["eta0"]
        self.strategy = solver_config["DM"]["strategy"]
        self.hyper_op = solver_config["hyper_op"]
        self.gda_loss = solver_config["gda_loss"]

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
            assert self.strategy in [
                "s1",
                "s2",
                "s3",
            ], "Three strategies are supported for DM operation, including ['s1','s2','s3']."
            if self.strategy == "s1":
                self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
                self.eta = (
                    (current_iter + 1) ** (-0.5 * self.tau)
                    * self.alpha**2
                    * self.ll_opt.defaults["lr"]
                )
                x_lr = (
                    (current_iter + 1) ** (-1.5 * self.tau)
                    * self.alpha**7
                    * self.ll_opt.defaults["lr"]
                )
            elif self.strategy == "s2":
                self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
                self.eta = (
                    (current_iter + 1) ** (-0.5 * self.tau)
                    * self.alpha
                    * self.ll_opt.defaults["lr"]
                )
                x_lr = (
                    (current_iter + 1) ** (-1.5 * self.tau)
                    * self.alpha**5
                    * self.ll_opt.defaults["lr"]
                )
            elif self.strategy == "s3":
                self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
                self.eta = (current_iter + 1) ** (
                    -0.5 * self.tau
                ) * self.ll_opt.defaults["lr"]
                x_lr = (
                    (current_iter + 1) ** (-1.5 * self.tau)
                    * self.alpha**3
                    * self.ll_opt.defaults["lr"]
                )
            for params in self.ul_opt.param_groups:
                params["lr"] = x_lr
        else:
            gda_loss = None
            assert (
                self.strategy == "s1"
            ), "Only 's1' strategy is supported for DM without GDA operation."

            x_lr = (
                self.ul_opt.defaults["lr"]
                * (current_iter + 1) ** (-self.tau)
                * self.ll_opt.defaults["lr"]
            )
            eta = (
                self.eta
                * (current_iter + 1) ** (-0.5 * self.tau)
                * self.ll_opt.defaults["lr"]
            )
            for params in self.auxiliary_v_opt.param_groups:
                params["lr"] = eta
            for params in self.ul_opt.param_groups:
                params["lr"] = x_lr
        #############
        # self.ll_opt.zero_grad()
        # self.auxiliary_v_opt.zero_grad()
        # loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
        # upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        # assert (self.alpha>0) and (self.alpha<1), \
        #         "Set the coefficient alpha properly in (0,1)."
        # loss_full = (1.0 - self.alpha) * loss_f + self.alpha * upper_loss

        if gda_loss is not None:
            ll_feed_dict["alpha"] = self.alpha
            loss_full = self.gda_loss(
                ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
            )
        else:
            loss_full = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)

        grad_y_temp = jit.grad(
            loss_full, list(auxiliary_model.parameters()), retain_graph=True
        )
        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        grad_outer_params = grad_unused_zero(
            upper_loss, list(auxiliary_model.parameters()), retain_graph=True
        )
        grads_phi_params = grad_unused_zero(
            loss_full, list(auxiliary_model.parameters()), retain_graph=True
        )
        grads = custom_grad(
            grads_phi_params,
            list(self.ul_model.parameters()),
            self.auxiliary_v,
            retain_graph=True,
        )  # dx (dy f) v
        grad_outer_hparams = grad_unused_zero(
            upper_loss, list(self.ul_model.parameters())
        )

        if "RAD" in self.hyper_op:
            vsp = custom_grad(
                grads_phi_params,
                list(auxiliary_model.parameters()),
                grad_outputs=self.auxiliary_v,
            )  # dy (dy f) v=d2y f v

            for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params):
                v0._custom_grad = v - gow
            update_tensor_grads(list(self.ll_model.parameters()), grad_y_temp)
            # self.ll_opt.step()
            manual_update(self.ll_opt, list(self.ll_model.parameters()))
            # self.auxiliary_v_opt.step()
            manual_update(self.auxiliary_v_opt, self.auxiliary_v)

            grads = [
                -g + v if g is not None else v
                for g, v in zip(grads, grad_outer_hparams)
            ]
            update_tensor_grads(list(self.ul_model.parameters()), grads)

        else:
            vsp = custom_grad(
                grads_phi_params,
                list(auxiliary_model.parameters()),
                grad_outputs=self.auxiliary_v,
            )

            tem = [v - gow for v, gow in zip(vsp, grad_outer_params)]

            ita_u = list_tensor_norm(tem) ** 2
            grad_tem = custom_grad(
                grads_phi_params, list(auxiliary_model.parameters()), grad_outputs=tem
            )
            ita_l = list_tensor_matmul(tem, grad_tem)

            ita = ita_u / (ita_l + 1e-12)

            self.auxiliary_v = [
                v0 - ita * v + ita * gow
                for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params)
            ]

            vsp = custom_grad(
                grads_phi_params,
                list(auxiliary_model.parameters()),
                grad_outputs=self.auxiliary_v,
            )

            for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params):
                v0._custom_grad = v - gow

            update_tensor_grads(list(self.ll_model.parameters()), grad_y_temp)
            manual_update(self.ll_opt, list(self.ll_model.parameters()))

            grads = [
                -g + v if g is not None else v
                for g, v in zip(grads, grad_outer_hparams)
            ]
            update_tensor_grads(list(self.ul_model.parameters()), grads)

        return -1
