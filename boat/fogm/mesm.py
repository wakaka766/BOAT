from boat.utils.op_utils import l2_reg
from ..dynamic_ol.dynamical_system import Dynamical_System
from boat.utils.op_utils import update_grads,grad_unused_zero,update_tensor_grads,copy_parameter_from_list
import numpy
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Callable
import copy
from typing import Dict, Any, Callable,List

class MESM(Dynamical_System):
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
        super(MESM, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.ul_objective = ul_objective
        self.ll_opt = ll_opt
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_loop = lower_loop
        self.eta = solver_config['MESM']["eta"]
        self.gamma_1 = solver_config['MESM']['gamma_1']
        self.c0 = solver_config['MESM']['c0']
        self.y_hat = copy.deepcopy(self.ll_model)
        self.y_hat_opt = torch.optim.SGD(self.y_hat.parameters(), lr=solver_config['MESM']['y_hat_lr'], momentum=0.9)
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

        if current_iter == 0:
            ck = 0.2
        else:
            ck = numpy.power(current_iter + 1, 0.25) * self.c0
        # params = [w.detach().requires_grad_(True) for w in list(self.ll_model.parameters())]  # y
        # last_param_theta = inner_loop(x.parameters(), vs, inner_opt, 1, log_interval=10)
        theta_loss = self.ll_objective(ll_feed_dict,self.ul_model,self.y_hat)

        grad_theta_parmaters = grad_unused_zero(theta_loss, list(self.y_hat.parameters()))

        ## some convert
        errs = []
        for a, b in zip(list(self.y_hat.parameters()), list(self.ll_model.parameters())):
            diff = a - b
            errs.append(diff)
        vs_param = []
        for v0, gt, err in zip(list(self.y_hat.parameters()), grad_theta_parmaters, errs):
            vs_param.append(v0 - self.eta * (gt + self.gamma_1 * err) )  # upate \theta

        copy_parameter_from_list(self.y_hat,vs_param)

        reg = 0
        for param1, param2 in zip(list(self.ll_model.parameters()), vs_param):
            diff = param1 - param2
            # result_params.append(diff)
            reg += torch.norm(diff, p=2) ** 2
        lower_loss = (1 / ck) * self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model) + self.ll_objective(ll_feed_dict,self.ul_model, self.ll_model) - 0.5 * self.gamma_1 * reg

        ## 直接对y求导
        # lower_loss = low_loss_FO(vs, params, x.parameters(), ck, gamma_1)

        self.ll_opt.zero_grad()
        grad_y_parmaters = grad_unused_zero(lower_loss, list(self.ll_model.parameters()))

        update_tensor_grads(self.ll_var, grad_y_parmaters)

        self.ll_opt.step()
        # copy_parameter_from_list(y, last_param[-1])

        upper_loss = (1 / ck) * self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model) + self.ll_objective(ll_feed_dict,self.ul_model, self.ll_model) - self.ll_objective(ll_feed_dict,self.ul_model, self.y_hat)
        grad_x_parmaters = grad_unused_zero(upper_loss, list(self.ul_model.parameters()))
        update_tensor_grads(self.ul_var, grad_x_parmaters)

        return upper_loss