import torch
from .hyper_gradient import HyperGradient
from ..utils.op_utils import update_grads,update_tensor_grads

from torch.nn import Module
from torch import Tensor
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase


class IGA(HyperGradient):
    r"""UL Variable Gradients Calculation with BSG Method

    Implements the UL problem optimization procedure of approximated Bilevel Stochastic
    Gradient method (BSG-1)`[1]`_, which approximates second-order UL gradient
    $$
    \nabla F = \nabla_xF - \nabla_{xy}^2f(\nabla_{yy}*2f)_{-1}\nabla_yF
    $$
    to first-order
    $$
    \nabla_xF-\frac{\nabla_yf^\top\nabla_yF}{\nabla_yf^\top\nabla_yf}\nabla_xf
    $$

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

        ll_objective: callable
            An optimization problem which is considered as the constraint of upper
            level problem.

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

        ll_model: Module
            The adapt_model whose parameters will be updated during upper-level optimization.

    References
    ----------
    _`[1]` T. Giovannelli, G. Kent, L. N. Vicente, "Bilevel stochastic methods for
    optimization and machine learning: Bilevel stochastic descent and DARTS", in arxiv, 2021.
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
        super(IGA, self).__init__(ul_objective, ul_model, ll_model,ll_var,ul_var)
        self.ll_objective = ll_objective
        self.alpha = solver_config['GDA']["alpha_init"]
        self.alpha_decay = solver_config['GDA']["alpha_decay"]
        self.gda_loss = solver_config['gda_loss']
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

            train_data: Tensor
                The training data used for upper level problem optimization.

            train_target: Tensor
                The labels of the samples in the train data.

        Returns
        -------
        upper_loss: Tensor
            Returns the loss value of upper objective.
        """
        if self.gda_loss is not None:
            ll_feed_dict['alpha'] = self.alpha*self.alpha_decay**max_loss_iter
            lower_loss = self.gda_loss(ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model)
        else:
            lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
        dfy = torch.autograd.grad(lower_loss, list(auxiliary_model.parameters()), retain_graph=True)

        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        dFy = torch.autograd.grad(upper_loss, list(auxiliary_model.parameters()), retain_graph=True)

        # calculate GN loss
        gFyfy = 0
        gfyfy = 0
        for Fy, fy in zip(dFy, dfy):
            gFyfy = gFyfy + torch.sum(Fy * fy)
            gfyfy = gfyfy + torch.sum(fy * fy)
        GN_loss = -gFyfy.detach() / gfyfy.detach() * lower_loss

        grads_upper = torch.autograd.grad(GN_loss + upper_loss, list(self.ul_var))
        update_tensor_grads( self.ul_var,grads_upper)

        return upper_loss
