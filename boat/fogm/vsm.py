from boat.utils.op_utils import l2_reg
from ..dynamic_ol.dynamical_system import DynamicalSystem
from boat.utils.op_utils import update_grads,update_tensor_grads
import torch
from torch.nn import Module
from torch.optim import Optimizer
import copy
from typing import Dict, Any, Callable,List


class VSM(DynamicalSystem):
    """
    Implements the optimization procedure of Value-function based Sequential (VSM) _`[1]`.

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
    :param ll_var: The list of lower-level variables of the BLO problem.
    :type ll_var: List
    :param ul_var: The list of upper-level variables of the BLO problem.
    :type ul_var: List
    :param lower_loop: Number of iterations for lower-level optimization.
    :type lower_loop: int
    :param solver_config: Dictionary containing solver configurations.
    :type solver_config: dict


    References
    ----------
    _`[1]` Liu B, Ye M, Wright S, et al. Bome! bilevel optimization made easy: A simple first-order approach[C].
    In NeurIPS, 2022.
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
        """
        Execute the optimization procedure with the data from feed_dict.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param current_iter: The current iteration number of the optimization process.
        :type current_iter: int

        :returns: None
        """
        reg_decay =  self.reg_decay * current_iter + 1
        for z_idx in range(self.z_loop):
            self.ll_opt.zero_grad()
            loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.parameters())
            loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            loss_z = loss_z_ + loss_l2_z
            grads = torch.autograd.grad(loss_z, list(self.ll_model.parameters()))
            update_grads(grads, self.ll_model)
            self.ll_opt.step()
        self.ll_opt.zero_grad()


        auxiliary_model = copy.deepcopy(self.ll_model)
        auxiliary_opt = torch.optim.SGD(auxiliary_model.parameters(), lr=self.z_lr)

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
        update_tensor_grads(self.ul_var, grads)
        return loss_x_
