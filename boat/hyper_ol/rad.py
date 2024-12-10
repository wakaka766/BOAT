import torch
from .hyper_gradient import HyperGradient
from ..utils.op_utils import update_grads,update_tensor_grads
from typing import List, Callable, Dict
from torch.nn import Module
from torch import Tensor
from higher.patch import _MonkeyPatchBase


class RAD(HyperGradient):
    r"""UL Variable Gradients Calculation with Reverse-mode AD

    Implements the UL optimization procedure with Reverse-mode Auto Diff method_`[1]`_.

    A wrapper of lower adapt_model that has been optimized in the lower optimization will
    be used in this procedure.

    Parameters
    ----------
        ul_objective: callable
            The main optimization problem in a hierarchical optimization problem.

            Callable with signature callable(state). Defined based on modeling of
            the specific problem that need to be solved. Computing the loss of upper
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

        truncate_max_loss_iter: bool, default=False
            Optional argument, if set True then during ul optimization IAPTT-GM method will be used to
            truncate the trajectory.

        update_ll_model_init: bool, default=False
            If set True, the initial value of ll model will be updated after this iteration.

    References
    ----------
    _`[1]` L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel
     programming for hyperparameter optimization and meta-learning", in ICML, 2018.

    _`[2]` R. Liu, Y. Liu, S. Zeng, J. Zhang, "Towards Gradient-based Bilevel
     Optimization with Non-convex Followers and Beyond", in NeurIPS, 2021.
    """

    def __init__(
            self,
            ll_objective: Callable,
            ul_objective: Callable,
            ll_model: Module,
            ul_model: Module,
            ll_var:List,
            ul_var:List,
            solver_config : Dict
    ):
        super(RAD, self).__init__(ul_objective, ul_model, ll_model,ll_var,ul_var)
        self.dynamic_initialization = "DI" in solver_config['dynamic_op']

    def compute_gradients(
            self,
            ll_feed_dict: Dict,
            ul_feed_dict: Dict,
            auxiliary_model: _MonkeyPatchBase,
            max_loss_iter: int = 0
    ):
        """
        Compute the grads of upper variable with validation data samples in the batch
        using upper objective. The grads will be saved in the passed in upper adapt_model.

        Note that the implemented UL optimization procedure will only compute
        the grads of upper variables。 If the validation data passed in is only single data
        of the batch (such as few-shot learning experiment), then compute_gradients()
        function should be called repeatedly to accumulate the grads of upper variables
        for the whole batch. After that the update operation of upper variables needs
        to be done outside this module.

        Parameters
        ----------
            validate_data: Tensor
               The validation data used for upper level problem optimization.

            validate_target: Tensor
               The labels of the samples in the validation data.

            auxiliary_model: _MonkeyPatchBase
                Wrapper of lower adapt_model encapsulated by module higher, has been optimized in lower
                optimization phase.

            max_loss_iter: int = 0
                The step of lower optimization loop which has the maximum loss value. The
                backpropagation trajectory shall be truncated here rather than the whole
                lower-loop.

        Returns
        -------
        upper_loss: Tensor
           Returns the loss value of upper objective.
        """
        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model,params=auxiliary_model.parameters(time=max_loss_iter))
        grads_upper = torch.autograd.grad(upper_loss, self.ul_model.parameters(),
                                          retain_graph=self.dynamic_initialization,allow_unused=True,materialize_grads=True)
        update_tensor_grads(self.ul_var,grads_upper)

        if self.dynamic_initialization:
            grads_lower = torch.autograd.grad(upper_loss, list(auxiliary_model.parameters(time=0)))
            update_tensor_grads(self.ll_var, grads_lower)

        return upper_loss
