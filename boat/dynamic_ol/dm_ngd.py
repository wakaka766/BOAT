from .dynamical_system import Dynamical_System
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Callable
from higher.patch import _MonkeyPatchBase
from higher.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from ..utils.op_utils import update_tensor_grads,grad_unused_zero,list_tensor_norm,list_tensor_matmul
import copy
class DM_NGD(Dynamical_System):

    r"""Lower adapt_model optimization procedure

    Implements the LL problem optimization procedure of two explicit gradient
    based methods (EGBMs) with lower-level singleton (LLS) assumption, Reverse-mode
    AutoDiff method (Recurrence) `[1]`_ and Truncated Recurrence method (T-Recurrence) `[2]`_, as well as
    two methods without LLS, Bi-level descent aggregation (BDA) `[3]`_ and Initialization
     Auxiliary and Pessimistic Trajectory Truncated Gradient Method (IAPTT-GM) `[4]`_.

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

        lower_loop: int
            Updating iterations over lower level optimization.

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

        truncate_max_loss_iter: bool, default=False
            Optional argument, if set True then during ul optimization IAPTT-GM method will be used to
            truncate the trajectory.

        alpha: float, default=0
            The aggregation parameter for Bi-level descent aggregation method, where
            alpha ∈ (0, 1) denotes the ratio of lower objective to upper objective
            during lower optimizing.

        alpha_decay: float, default=0.0
            Weight decay coefficient of aggregation parameter alpha. The decay rate will accumulate
            with ll optimization procedure.

        truncate_iters: int, default=0
            Specific parameter for Truncated Reverse method, defining number of
            iterations to truncate in the back propagation process during lower
            optimizing.

        ll_opt: BOATOptimizer, default=None
            The original optimizer of lower adapt_model.

    References
    ----------
    _`[1]` L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel
     programming for hyperparameter optimization and meta-learning", in ICML, 2018.

    _`[2]` A. Shaban, C. Cheng, N. Hatch, and B. Boots, "Truncated backpropagation
     for bilevel optimization", in AISTATS, 2019.

    _`[3]` R. Liu, P. Mu, X. Yuan, S. Zeng, and J. Zhang, "A generic first-order algorithmic
     framework for bi-level programming beyond lower-level singleton", in ICML, 2020.

    _`[4]` R. Liu, Y. Liu, S. Zeng, and J. Zhang, "Towards Gradient-based Bilevel
     Optimization with Non-convex Followers and Beyond", in NeurIPS, 2021.
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

        super(DM_NGD, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
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
        Execute the lower optimization procedure with training data samples using lower
        objective. The passed in wrapper of lower adapt_model will be updated.

        Parameters
        ----------
            train_data: Tensor
                The training data used for LL problem optimization.

            train_target: Tensor
                The labels of the samples in the train data.

            auxiliary_model: _MonkeyPatchBase
                Wrapper of lower adapt_model encapsulated by module higher, will be optimized in lower
                optimization procedure.

            auxiliary_opt: DifferentiableOptimizer
                Wrapper of lower optimizer encapsulated by module higher, will be used in lower
                optimization procedure.

            validate_data: Tensor (optional), default=None
                The validation data used for UL problem optimization. Needed when using BDA
                method or IAPTT-GM method.

            validate_target: Tensor (optional), default=None
                The labels of the samples in the validation data. Needed when using
                BDA method or IAPTT-GM method.

        Returns
        -------
        None or int
            If use IAPTT-GM method as upper level optimization method, then will return
            the num of iter which has the maximum loss value among the entire iterative
            procedure of lower level optimization. Otherwise will return None.
        """
        assert self.strategy == "s1", \
                "Only 's1' strategy is supported for DM without GDA operation."

        x_lr = self.ul_lr * (current_iter + 1) ** (-self.tau) * self.ll_opt.defaults['lr']
        eta = self.eta * (current_iter + 1) ** (-0.5 * self.tau) * self.ll_opt.defaults['lr']
        for params in self.auxiliary_v_opt.param_groups:
            params['lr'] = eta
        for params in self.ul_opt.param_groups:
            params['lr'] = x_lr

        #############
        self.ll_opt.zero_grad()
        self.auxiliary_v_opt.zero_grad()
        loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
        grad_y_temp = torch.autograd.grad(loss_f,auxiliary_model.parameters(),retain_graph=True)

        #############

        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        grad_outer_params = grad_unused_zero(upper_loss, list(auxiliary_model.parameters()))
        grads_phi_params = grad_unused_zero(loss_f, list(auxiliary_model.parameters()),create_graph=True,retain_graph=True)
        grads = grad_unused_zero(grads_phi_params, list(self.ul_model.parameters()), grad_outputs=self.auxiliary_v,retain_graph=True)  # dx (dy f) v
        grad_outer_hparams =grad_unused_zero(upper_loss, list(self.ul_model.parameters()))

        if "RAD" in self.hyper_op:
            vsp = grad_unused_zero(grads_phi_params, list(auxiliary_model.parameters()), grad_outputs=self.auxiliary_v)  # dy (dy f) v=d2y f v

            for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params):
                v0.grad = v - gow
            update_tensor_grads(list(self.ll_model.parameters()), grad_y_temp)
            self.ll_opt.step()
            self.auxiliary_v_opt.step()

            grads = [-g + v for g, v in zip(grads, grad_outer_hparams)]
            update_tensor_grads(list(self.ul_model.parameters()), grads)
        else:
            vsp = torch.autograd.grad(grads_phi_params, list(auxiliary_model.parameters()), grad_outputs=self.auxiliary_v, retain_graph=True,allow_unused=True)  # dy (dy f) v=d2y f v
            tem = [v - gow for v, gow in zip(vsp, grad_outer_params)]

            ita_u = list_tensor_norm(tem) ** 2
            grad_tem = torch.autograd.grad(grads_phi_params, list(auxiliary_model.parameters()), grad_outputs=tem, retain_graph=True,
                                  allow_unused=True)  # dy (dy f) v=d2y f v

            ita_l = list_tensor_matmul(tem, grad_tem, trans=1)
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

