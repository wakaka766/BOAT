import torch.autograd

from .dynamical_system import Dynamical_System

from torch.nn import Module
from torch import Tensor
from typing import Callable
from higher.patch import _MonkeyPatchBase
from higher.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from ..utils.op_utils import stop_grads
class NGD(Dynamical_System):
    r"""Lower level model optimization procedure

    Implement the LL model update process.

    The implemented lower level optimization procedure will optimize a wrapper of lower
     level model for further using in the following upper level optimization.

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

        ll_model: Module
            Lower adapt_model in a hierarchical adapt_model structure whose parameters will be
            updated with lower objective during lower-level optimization.
    """

    def __init__(
            self,
            ll_objective: Callable,
            ul_objective: Callable,
            ll_model: Module,
            ul_model: Module,
            lower_loop: int,
            solver_config: Dict[str, Any]
    ):

        super(NGD, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]
        self.truncate_iters = solver_config['RGT']["truncate_iter"]
        self.ul_objective=ul_objective
        self.ll_opt = solver_config['ll_opt']
        self.foa = 'FOA' in solver_config['hyper_op']
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

        Returns
        -------
        None
        """

        if self.truncate_iters > 0:
            ll_backup = [x.data.clone().detach().requires_grad_() for x in self.ll_model.parameters()]
            for lower_iter in range(self.truncate_iters):
                lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
                lower_loss.backward()
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
                lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
                auxiliary_opt.step(lower_loss)
                upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
                # print(torch.autograd.grad(upper_loss,list(self.ul_model.parameters()),allow_unused=False))
                ul_loss_list.append(upper_loss.item())
            ll_step_with_max_ul_loss = ul_loss_list.index(max(ul_loss_list))
            return ll_step_with_max_ul_loss+1
        for lower_iter in range(self.lower_loop - self.truncate_iters):
            lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
            auxiliary_opt.step(lower_loss,grad_callback= stop_grads if self.foa else None)
        return self.lower_loop - self.truncate_iters