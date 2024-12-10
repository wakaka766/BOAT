from boat.utils.op_utils import l2_reg
from ..dynamic_ol.dynamical_system import Dynamical_System
from boat.utils.op_utils import update_grads,grad_unused_zero,require_model_grad,update_tensor_grads,stop_model_grad

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Callable
import copy
from typing import Dict, Any, Callable,List

class VFM(Dynamical_System):
    r"""Lower adapt_model optimization procedure of Value-Function-based Interior-point Method

    Implements the LL problem optimization procedure of Value-Function Best-
    Response (VFBR) type BLO methods, named i-level Value-Function-basedInterior-point
    Method(BVFIM) `[1]`_.

    The implemented lower level optimization procedure will optimize a wrapper of lower
    adapt_model for further using in the following upper level optimization.

    Parameters
    ----------
        ll_objective: callable
            An optimization problem which is considered as the constraint of upper
            level problem.

            Callable with signature callable(state). Defined based on modeling of
            the specific problem that need to be solved. Computing the loss of LL
            problem. The state object contains the following:

            - "data"
                Data used in the upper optimization phase.
            - "target"
                Target used in the upper optimization phase.
            - "ul_model"
                Upper adapt_model of the bi-level adapt_model structure.
            - "ll_model"
                Lower adapt_model of the bi-level adapt_model structure.

        ul_model: Module
            Upper adapt_model in a hierarchical adapt_model structure whose parameters will be
            updated with upper objective.

        ul_objective: callable
            The main optimization problem in a hierarchical optimization problem.

            Callable with signature callable(state). Defined based on modeling of
            the specific problem that need to be solved. Computing the loss of UL
            problem. The state object contains the following:

            - "data"
                Data used in the upper optimization phase.
            - "target"
                Target used in the upper optimization phase.
            - "ul_model"
                Upper adapt_model of the bi-level adapt_model structure.
            - "ll_model"
                Lower adapt_model of the bi-level adapt_model structure.

        ll_model: Module
            Lower adapt_model in a hierarchical adapt_model structure whose parameters will be
            updated with lower objective during lower-level optimization.

        lower_loop: int, default=5
            Num of steps to obtain a low LL problem value, i.e. optimize LL variable
            with LL problem. Regarded as $T_z$ in the paper.

        y_loop: int, default=5
            Num of steps to obtain a optimal LL variable under the LL problem value obtained
            after z_loop, i.e. optimize the updated LL variable with UL problem. Regarded as
            Regarded as $T_y$ in the paper.

        ll_l2_reg: float, default=0.1
            Weight of L2 regularization term in the value function of the regularized
            LL problem, which is $\displaystyle f_\mu^* = \min_{y\in\mathbb{R}^n}
            f(x,y) + \frac{\mu_1}{2}\|y\|^2 + \mu_2$.

        ul_l2_reg: float, default=0.01
            Weight of L2 regularization term in the value function of the regularized
            UL problem, which is $\displaystyle \varphi(x) = \min_{y\in\mathbb{R}^n} F(x,y)
             + \frac{\theta}{2}\|y\|^2 - \tau\ln(f_\mu^*(x)-f(x,y))$.
.

        ul_ln_reg: float, default=10.
            Weight of the log-barrier penalty term in the value function of the regularized
            UL problem, as ul_l2_reg.

    References
    ----------
    _`[1]` R. Liu, X. Liu, X. Yuan, S. Zeng and J. Zhang, "A Value-Function-based
    Interior-point Method for Non-convex Bi-level Optimization", in ICML, 2021.
    """
    def __init__(
            self,
            ll_objective: Callable,
            lower_loop: int,
            ul_model: Module,
            ul_objective: Callable,
            ll_model: Module,
            ll_opt: Optimizer,
            ll_var: List,
            ul_var: List,
            solver_config: Dict[str, Any]
    ):
        super(VFM, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.ul_objective = ul_objective
        self.ll_opt = ll_opt
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_hat_lr = float(solver_config['VFM']['y_hat_lr'])
        self.eta = solver_config['VFM']["eta"]
        self.u1 = solver_config['VFM']["u1"]
        self.device = solver_config["device"]
    def optimize(
            self,
            ll_feed_dict: Dict,
            ul_feed_dict: Dict,
            current_iter: int
    ):
        r"""
        Execute the lower optimization procedure with training data samples using lower
        objective. The passed in wrapper of lower adapt_model will be updated.

        Parameters
        ----------
            train_data: Tensor
                The training data used for LL problem optimization.

            train_target: Tensor
                The labels of the samples in the train data.

            auxiliary_model: Module
                Wrapper of lower adapt_model encapsulated by module higher, will be optimized in lower
                optimization procedure.  # todo

            auxiliary_opt: Optimizer
                Wrapper of lower optimizer encapsulated by module higher, will be used in lower
                optimization procedure.  # todo

            validate_data:Tensor
                The validation data used for UL problem.

            validate_target: Tensor
                The labels of the samples in the validation data.

            reg_decay: float
                Weight decay coefficient of L2 regularization term and log-barrier
                penalty term.The value increases with the number of iterations.
        """
        y_hat = copy.deepcopy(self.ll_model)
        y_hat_opt = torch.optim.SGD(y_hat.parameters(), lr=self.y_hat_lr, momentum=0.9)
        n_params_y = sum([p.numel() for p in self.ll_model.parameters()])
        n_params_x = sum([p.numel() for p in self.ul_model.parameters()])
        delta_f = torch.zeros(n_params_x + n_params_y).to(self.device)
        delta_F = torch.zeros(n_params_x + n_params_y).to(self.device)

        def g_x_xhat_w(y, y_hat, x):
            loss = self.ll_objective(ll_feed_dict, x, y) - self.ll_objective(ll_feed_dict, x, y_hat)
            grad_y = grad_unused_zero(loss, list(y.parameters()), retain_graph=True)
            grad_x = grad_unused_zero(loss, list(x.parameters()))
            return loss, grad_y, grad_x

        require_model_grad(y_hat)
        for y_itr in range(self.lower_loop):
            y_hat_opt.zero_grad()
            tr_loss = self.ll_objective(ll_feed_dict, self.ul_model, y_hat)
            grads_hat = torch.autograd.grad(tr_loss, y_hat.parameters(), allow_unused=True)
            update_tensor_grads(list(y_hat.parameters()), grads_hat)
            y_hat_opt.step()
        F_y = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)

        grad_F_y = grad_unused_zero(F_y, list(self.ll_model.parameters()), retain_graph=True)
        grad_F_x = grad_unused_zero(F_y, list(self.ul_model.parameters()))
        stop_model_grad(y_hat)
        loss, gy, gx_minus_gx_k = g_x_xhat_w(self.ll_model, y_hat, self.ul_model)
        delta_F[:n_params_y].copy_(torch.cat([fc_param.view(-1).clone() for fc_param in grad_F_y]).view(-1).clone())
        delta_f[:n_params_y].copy_(torch.cat([fc_param.view(-1).clone() for fc_param in gy]).view(-1).clone())
        delta_F[n_params_y:].copy_(torch.cat([fc_param.view(-1).clone() for fc_param in grad_F_x]).view(-1).clone())
        delta_f[n_params_y:].copy_(torch.cat([fc_param.view(-1).clone() for fc_param in gx_minus_gx_k]).view(-1).clone())
        norm_dq = delta_f.norm().pow(2)
        dot = delta_F.dot(delta_f)
        d = delta_F + F.relu((self.u1*loss-dot)/(norm_dq+1e-8)) * delta_f
        y_grad = []
        x_grad = []
        all_numel = 0
        for _, param in enumerate(self.ll_model.parameters()):
            y_grad.append((d[all_numel:all_numel + param.numel()]).data.view(tuple(param.shape)).clone())
            all_numel = all_numel + param.numel()
        for _, param in enumerate(self.ul_model.parameters()):
            x_grad.append((d[all_numel:all_numel + param.numel()]).data.view(tuple(param.shape)).clone())
            all_numel = all_numel + param.numel()


        update_tensor_grads(self.ll_var, y_grad)
        update_tensor_grads(self.ul_var, x_grad)
        # for _, param in enumerate(self.):
        #     param.grad = y_grad[_]
        # for _, param in enumerate(self.ul_model.parameters()):
        #     param.grad = x_grad[_]
        self.ll_opt.step()
        return F_y