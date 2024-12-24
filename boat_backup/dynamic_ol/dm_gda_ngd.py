from .dynamical_system import DynamicalSystem
import torch
from torch.nn import Module
from ..higher_ms.patch import _MonkeyPatchBase
from ..higher_ms.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from ..utils.op_utils import update_tensor_grads,grad_unused_zero,list_tensor_norm,list_tensor_matmul

class DM_GDA_NGD(DynamicalSystem):

    """
    Implements the lower-level optimization procedure of the Naive Gradient Descent (NGD) _`[1]`, Gradient Descent
    Aggregation (GDA) _`[2]` and Dual Multiplier (DM) _`[3]`.

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

    _`[3]` Liu R, Liu Y, Yao W, et al. Averaged method of multipliers for bi-level optimization without lower-level
    strong convexity [C]. In ICML, 2023.
    """

    def __init__(
            self,
            ll_objective: Callable,
            lower_loop: int,
            ul_model: Module,
            ul_objective: Callable,
            ll_model: Module,
            solver_config: Dict[str, Any]
    ):

        super(DM_GDA_NGD, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]
        self.ul_objective = ul_objective
        self.alpha = solver_config['GDA']["alpha_init"]
        self.alpha_decay = solver_config['GDA']["alpha_decay"]
        self.truncate_iters = solver_config['RGT']["truncate_iter"]
        self.ll_opt = solver_config['ll_opt']
        self.auxiliary_v = solver_config["DM"]['auxiliary_v']
        self.auxiliary_v_opt = solver_config["DM"]['auxiliary_v_opt']
        self.auxiliary_v_lr = solver_config["DM"]['auxiliary_v_lr']
        self.tau = solver_config['DM']['tau']
        self.p = solver_config['DM']['p']
        self.mu0 = solver_config['DM']['mu0']
        self.eta = solver_config['DM']['eta0']
        self.strategy = solver_config['DM']['strategy']
        self.hyper_op =  solver_config["hyper_op"]

    def optimize(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        auxiliary_opt: DifferentiableOptimizer,
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

        :param auxiliary_model: A patched lower model wrapped by the `higher_ms` library.
            It serves as the lower-level model for optimization.
        :type auxiliary_model: _MonkeyPatchBase

        :param auxiliary_opt: A patched optimizer for the lower-level model,
            wrapped by the `higher` library. This optimizer allows for differentiable optimization.
        :type auxiliary_opt: DifferentiableOptimizer

        :param current_iter: The current iteration number of the optimization process.
        :type current_iter: int

        :returns: None
        """
        assert self.strategy in ["s1","s2","s3"], \
                "Three strategies are supported for DM operation, including ['s1','s2','s3']."
        if self.strategy == 's1':
            self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
            self.eta = (current_iter + 1) ** (-0.5 * self.tau) * self.alpha**2 * self.ll_opt.defaults['lr']
            x_lr = (current_iter + 1) ** (-1.5 * self.tau) * self.alpha ** 7 * self.ll_opt.defaults['lr']
        elif self.strategy == 's2':
            self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
            self.eta = (current_iter + 1) ** (-0.5 * self.tau) * self.alpha* self.ll_opt.defaults['lr']
            x_lr = (current_iter + 1) ** (-1.5 * self.tau) * self.alpha ** 5 * self.ll_opt.defaults['lr']
        elif self.strategy == 's3':
            self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
            self.eta = (current_iter + 1) ** (-0.5 * self.tau)  * self.ll_opt.defaults['lr']
            x_lr = (current_iter + 1) ** (-1.5 * self.tau) * self.alpha ** 3 * self.ll_opt.defaults['lr']
        for params in self.ul_opt.param_groups:
            params['lr'] = x_lr
        #############
        self.ll_opt.zero_grad()
        self.auxiliary_v_opt.zero_grad()
        loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        assert (self.alpha>0) and (self.alpha<1), \
                "Set the coefficient alpha properly in (0,1)."
        loss_full = (1.0 - self.alpha) * loss_f + self.alpha * upper_loss
        grad_y_temp = torch.autograd.grad(loss_full, auxiliary_model.parameters(),retain_graph=True)

        grad_outer_params = grad_unused_zero(upper_loss, list(auxiliary_model.parameters()),retain_graph=True)
        grads_phi_params = grad_unused_zero(loss_full, list(auxiliary_model.parameters()),create_graph=True,retain_graph=True)
        grads = grad_unused_zero(grads_phi_params, list(self.ul_model.parameters()), grad_outputs=self.auxiliary_v,retain_graph=True)  # dx (dy f) v
        grad_outer_hparams = grad_unused_zero(upper_loss, list(self.ul_model.parameters()))

        if "RAD" in self.hyper_op:
            vsp = grad_unused_zero(grads_phi_params, list(auxiliary_model.parameters()),
                                   grad_outputs=self.auxiliary_v)  # dy (dy f) v=d2y f v

            for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params):
                v0.grad = v - gow
            update_tensor_grads(list(self.ll_model.parameters()), grad_y_temp)
            self.ll_opt.step()
            self.auxiliary_v_opt.step()

            grads = [-g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]
            update_tensor_grads(list(self.ul_model.parameters()), grads)

        else:

            vsp = torch.autograd.grad(grads_phi_params, list(auxiliary_model.parameters()), grad_outputs=self.auxiliary_v, retain_graph=True,allow_unused=True)  # dy (dy f) v=d2y f v
            tem = [v - gow for v, gow in zip(vsp, grad_outer_params)]

            ita_u = list_tensor_norm(tem) ** 2
            grad_tem = torch.autograd.grad(grads_phi_params, list(auxiliary_model.parameters()), grad_outputs=tem, retain_graph=True,
                                  allow_unused=True)  # dy (dy f) v=d2y f v

            ita_l = list_tensor_matmul(tem, grad_tem)
            # print(ita_u,ita_l)
            ita = ita_u / (ita_l + 1e-12)
            self.auxiliary_v = [v0 - ita * v + ita * gow for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params)]  # (I-ita*d2yf)v+ita*dy F)

            vsp = torch.autograd.grad(grads_phi_params, list(auxiliary_model.parameters()), grad_outputs=self.auxiliary_v,
                             allow_unused=True)  # dy (dy f) v=d2y f v

            for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params):
                v0.grad = v - gow
            update_tensor_grads(list(self.ll_model.parameters()), grad_y_temp)
            self.ll_opt.step()

            grads = [-g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]
            update_tensor_grads(list(self.ul_model.parameters()), grads)

        return -1
        # if self.alpha > 0.0 or self.truncate_max_loss_iter:
        #     assert ul_feed_dict is not None,\
        #         "GDA and PTT method need validate data and validate target"
        # alpha = self.alpha

        # truncate with T-RAD method
        # if self.truncate_iters > 0:
        #     ll_backup = [x.data.clone().detach().requires_grad_() for x in self.ll_model.parameters()]
        #     for lower_iter in range(self.truncate_iters):
        #         lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        #         if self.alpha == 0.0:
        #             loss_f = lower_loss
        #         else:
        #             upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        #             loss_f = (1.0 - alpha) * lower_loss + alpha * upper_loss
        #         loss_f.backward()
        #         self.ll_opt.step()
        #         self.ll_opt.zero_grad()
        #     for x, y in zip(self.ll_model.parameters(), auxiliary_model.parameters()):
        #         y.data = x.data.clone().detach().requires_grad_()
        #     for x, y in zip(ll_backup, self.ll_model.parameters()):
        #         y.data = x.data.clone().detach().requires_grad_()
        #
        #
        # # truncate with PTT method
        # if self.truncate_max_loss_iter:
        #     ul_loss_list = []
        #     for lower_iter in range(self.lower_loop):
        #         lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
        #         if self.alpha == 0.0:
        #             auxiliary_opt.step(lower_loss)
        #         else:
        #             upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        #             loss_f = (1.0 - alpha) * lower_loss + alpha * upper_loss
        #             auxiliary_opt.step(loss_f)
        #             alpha = alpha * self.alpha_decay
        #         upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        #         ul_loss_list.append(upper_loss.item())
        #     ll_step_with_max_ul_loss = ul_loss_list.index(max(ul_loss_list))
        #     return ll_step_with_max_ul_loss+1
        #
        #
        #
        # for lower_iter in range(self.lower_loop - self.truncate_iters):
        #     lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
        #     if self.alpha == 0.0:
        #         auxiliary_opt.step(lower_loss)
        #     else:
        #         upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        #         loss_f = (1.0 - alpha) * lower_loss + alpha * upper_loss
        #         auxiliary_opt.step(loss_f)
        #         alpha = alpha * self.alpha_decay
