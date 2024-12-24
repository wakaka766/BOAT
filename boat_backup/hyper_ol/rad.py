import mindspore
from mindspore import nn, ops
from mindspore.common import Tensor
from .hyper_gradient import HyperGradient
from typing import List, Callable, Dict
from boat_ms.higher_ms.patch import _MonkeyPatchBase
from boat_ms.utils.op_utils import update_tensor_grads


class RAD(HyperGradient):
    """
    Calculation of the hyper gradient of the upper-level variables with Reverse Auto Differentiation (RAD) _`[1]`.

    Parameters
    ----------
        :param ll_objective: The lower-level objective of the BLO problem.
        :type ll_objective: callable
        :param ul_objective: The upper-level objective of the BLO problem.
        :type ul_objective: callable
        :param ll_model: The lower-level model of the BLO problem.
        :type ll_model: mindspore.nn.Cell
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: mindspore.nn.Cell
        :param ll_var: List of variables optimized with the lower-level objective.
        :type ll_var: List
        :param ul_var: List of variables optimized with the upper-level objective.
        :type ul_var: List
        :param solver_config: Dictionary containing solver configurations.
        :type solver_config: dict

    References
    ----------
    _`[1]` Franceschi, Luca, et al. Forward and reverse gradient-based hyperparameter optimization. in ICML, 2017.
    """

    def __init__(
            self,
            ll_objective: Callable,
            ul_objective: Callable,
            ll_model: nn.Cell,
            ul_model: nn.Cell,
            ll_var: List,
            ul_var: List,
            solver_config: Dict
    ):
        super(RAD, self).__init__(ul_objective, ul_model, ll_model, ll_var, ul_var)
        self.dynamic_initialization = "DI" in solver_config['dynamic_op']

    def compute_gradients(
            self,
            ll_feed_dict: Dict,
            ul_feed_dict: Dict,
            auxiliary_model: _MonkeyPatchBase,
            max_loss_iter: int = 0
    ):
        """
        Compute the hyper-gradients of the upper-level variables with the data from feed_dict and patched models.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param auxiliary_model: A patched lower model wrapped by the `higher_ms` library.
            It serves as the lower-level model for optimization.
        :type auxiliary_model: _MonkeyPatchBase

        :param max_loss_iter: The number of iteration used for backpropagation.
        :type max_loss_iter: int

        :returns: The current upper-level objective.
        """

        # Compute upper-level loss
        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model,
                                       params=auxiliary_model.parameters(time=max_loss_iter))

        # Compute gradients for upper-level variables
        grad_fn_upper = mindspore.value_and_grad(self.ul_objective, grad_position=(1, 2), weights=self.ul_var)
        grads_upper, _ = grad_fn_upper(ul_feed_dict, ll_feed_dict)
        update_tensor_grads(self.ul_var, grads_upper)

        # Optionally compute gradients for lower-level variables
        if self.dynamic_initialization:
            grad_fn_lower = mindspore.value_and_grad(self.ul_objective, grad_position=2,
                                                     weights=list(auxiliary_model.parameters(time=0)))
            grads_lower, _ = grad_fn_lower(ul_feed_dict, ll_feed_dict)
            update_tensor_grads(self.ll_var, grads_lower)

        return upper_loss
