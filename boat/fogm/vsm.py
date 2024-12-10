from boat.utils.op_utils import l2_reg
from ..dynamic_ol.dynamical_system import Dynamical_System
from boat.utils.op_utils import update_grads,update_tensor_grads
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Callable
import copy
from typing import Dict, Any, Callable,List

class VSM(Dynamical_System):
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
        super(VSM, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.ul_objective = ul_objective
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.ll_opt = ll_opt
        self.y_loop = lower_loop
        self.z_loop = solver_config['VSM']["z_loop"]
        self.ll_l2_reg = solver_config['VSM']["ll_l2_reg"]
        self.ul_l2_reg = solver_config['VSM']["ul_l2_reg"]
        self.ul_ln_reg = solver_config['VSM']["ul_ln_reg"]
        self.reg_decay = float(solver_config['VSM']['reg_decay'])
        self.z_lr = solver_config['VSM']['z_lr']

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
        reg_decay =  self.reg_decay * current_iter + 1
        for z_idx in range(self.z_loop):
            self.ll_opt.zero_grad()
            loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.parameters())
            loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            loss_z = loss_z_ + loss_l2_z
            grads = torch.autograd.grad(loss_z, list(self.ll_model.parameters()))
            update_grads(grads, self.ll_model)
            # loss_z.backward()
            self.ll_opt.step()
        self.ll_opt.zero_grad()

        # for x, y in zip(self.ll_model.parameters(), auxiliary_model.parameters()):
        #     y.data = x.data.clone().detach().requires_grad_()
        auxiliary_model = copy.deepcopy(self.ll_model)
        # auxiliary_opt = torch.optim.SGD(auxiliary_model.parameters(), lr=0.01)
        # auxiliary_opt = copy.deepcopy(self.ll_opt)
        auxiliary_opt = torch.optim.SGD(auxiliary_model.parameters(), lr=self.z_lr)
        # auxiliary_opt.param_groups[0]['params'] = list(auxiliary_model.parameters())

        with torch.no_grad():
            loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.parameters())
            loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            loss_z = loss_z_ + loss_l2_z

        for y_idx in range(self.y_loop):
            auxiliary_opt.zero_grad()
            loss_y_f_ = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
            loss_y_ = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
            loss_l2_y = l2_reg(auxiliary_model.parameters())
            loss_l2_y = self.ul_l2_reg / reg_decay * loss_l2_y
            loss_ln = torch.log(loss_y_f_.item() + loss_z.item() - loss_y_f_)
            loss_ln = self.ul_ln_reg / reg_decay * loss_ln
            loss_y = loss_y_ - loss_ln + loss_l2_y
            grads = torch.autograd.grad(loss_y, list(auxiliary_model.parameters()))
            update_grads(grads, auxiliary_model)
            # loss_y.backward()
            auxiliary_opt.step()
        auxiliary_opt.step()

        loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.parameters())
        loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        loss_z = loss_z_ + loss_l2_z

        loss_y_f_ = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
        loss_ln = self.ul_ln_reg / reg_decay * torch.log(loss_y_f_.item() + loss_z - loss_y_f_)

        loss_x_ = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        loss_x = loss_x_ - loss_ln
        grads = torch.autograd.grad(loss_x, list(self.ul_model.parameters()))

        # update_grads(grads, self.ul_model)
        update_tensor_grads(self.ul_var, grads)
        return loss_x_