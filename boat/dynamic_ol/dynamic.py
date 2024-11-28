from .dynamical_system import DynamicalSystem

from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Callable
from higher.patch import _MonkeyPatchBase
from higher.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable

class Dynamic(DynamicalSystem):
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

        super(Dynamic, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.truncate_max_loss_iter = solver_config['PTT']["truncate_max_loss_iter"]
        self.ul_objective = ul_objective
        self.alpha = solver_config['GDA']["alpha_init"]
        self.alpha_decay = solver_config['GDA']["alpha_decay"]
        self.truncate_iters = solver_config['RGT']["truncate_iter"]
        self.ll_opt = solver_config['ll_opt']

    def optimize(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        auxiliary_opt: DifferentiableOptimizer
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

        if self.alpha > 0.0 or self.truncate_max_loss_iter:
            assert ul_feed_dict is not None,\
                "GDA and PTT method need validate data and validate target"
        alpha = self.alpha

        # truncate with PTT method
        if self.truncate_max_loss_iter:
            ul_loss_list = []
            for lower_iter in range(self.lower_loop):
                lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
                if self.alpha == 0.0:
                    auxiliary_opt.step(lower_loss)
                else:
                    upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
                    loss_f = (1.0 - alpha) * lower_loss + alpha * upper_loss
                    auxiliary_opt.step(loss_f)
                    alpha = alpha * self.alpha_decay
                upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
                ul_loss_list.append(upper_loss.item())
            ll_step_with_max_ul_loss = ul_loss_list.index(max(ul_loss_list))
            return ll_step_with_max_ul_loss+1

        # truncate with T-RAD method
        if self.truncate_iters > 0:
            ll_backup = [x.data.clone().detach().requires_grad_() for x in self.ll_model.parameters()]
            for lower_iter in range(self.truncate_iters):
                lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
                if self.alpha == 0.0:
                    loss_f = lower_loss
                else:
                    upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
                    loss_f = (1.0 - alpha) * lower_loss + alpha * upper_loss
                loss_f.backward()
                self.ll_opt.step()
                self.ll_opt.zero_grad()
            for x, y in zip(self.ll_model.parameters(), auxiliary_model.parameters()):
                y.data = x.data.clone().detach().requires_grad_()
            for x, y in zip(ll_backup, self.ll_model.parameters()):
                y.data = x.data.clone().detach().requires_grad_()

        for lower_iter in range(self.lower_loop - self.truncate_iters):
            lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
            if self.alpha == 0.0:
                auxiliary_opt.step(lower_loss)
            else:
                upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
                loss_f = (1.0 - alpha) * lower_loss + alpha * upper_loss
                auxiliary_opt.step(loss_f)
                alpha = alpha * self.alpha_decay
