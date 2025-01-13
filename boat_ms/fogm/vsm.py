from boat_ms.utils.op_utils import l2_reg
import mindspore as ms
from typing import Dict, Any, Callable, List
from mindspore import nn, ops, Tensor, numpy as mnp
import copy
from boat_ms.operation_registry import register_class
from boat_ms.dynamic_ol.dynamical_system import DynamicalSystem


@register_class
class VSM(DynamicalSystem):
    """
    Implements the optimization procedure of Value-function based Sequential Method (VSM) [1].

    Parameters
    ----------
    ll_objective : Callable
        The lower-level objective function of the BLO problem.
    ul_objective : Callable
        The upper-level objective function of the BLO problem.
    ll_model : mindspore.nn.Cell
        The lower-level model of the BLO problem.
    ul_model : mindspore.nn.Cell
        The upper-level model of the BLO problem.
    ll_var : List[mindspore.Tensor]
        A list of lower-level variables of the BLO problem.
    ul_var : List[mindspore.Tensor]
        A list of upper-level variables of the BLO problem.
    lower_loop : int
        The number of iterations for lower-level optimization.
    solver_config : Dict[str, Any]
        A dictionary containing configurations for the solver. Expected keys include:

        - "lower_level_opt" (mindspore.nn.optim.Optimizer): Optimizer for the lower-level model.
        - "VSM" (Dict): Configuration for the VSM algorithm:
            - "z_loop" (int): Number of iterations for optimizing the auxiliary variable `z`.
            - "ll_l2_reg" (float): L2 regularization coefficient for the lower-level model.
            - "ul_l2_reg" (float): L2 regularization coefficient for the upper-level model.
            - "ul_ln_reg" (float): Logarithmic regularization coefficient for the upper-level model.
            - "reg_decay" (float): Decay rate for the regularization coefficients.
            - "z_lr" (float): Learning rate for optimizing the auxiliary variable `z`.
        - "device" (str): Device on which computations are performed, e.g., "cpu" or "cuda".

    References
    ----------
    [1] Liu B, Ye M, Wright S, et al. "BOME! Bilevel Optimization Made Easy: A Simple First-Order Approach," in NeurIPS, 2022.
    """

    def __init__(
        self,
        ll_objective: Callable,
        lower_loop: int,
        ul_model: nn.Cell,
        ul_objective: Callable,
        ll_model: nn.Cell,
        ll_var: List,
        ul_var: List,
        solver_config: Dict[str, Any],
    ):
        super(VSM, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ul_opt = solver_config["upper_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_loop = lower_loop
        self.z_loop = solver_config["VSM"]["z_loop"]
        self.ll_l2_reg = solver_config["VSM"]["ll_l2_reg"]
        self.ul_l2_reg = solver_config["VSM"]["ul_l2_reg"]
        self.ul_ln_reg = solver_config["VSM"]["ul_ln_reg"]
        self.reg_decay = float(solver_config["VSM"]["reg_decay"])
        self.z_lr = solver_config["VSM"]["z_lr"]

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        """
        Execute the optimization procedure with the data from feed_dict.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization. It typically includes training data, targets, and other information required to compute the LL objective.
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization. It typically includes validation data, targets, and other information required to compute the UL objective.
        current_iter : int
            The current iteration number of the optimization process.

        Returns
        -------
        The upper-level loss.
        """

        reg_decay = self.reg_decay * current_iter + 1

        for z_idx in range(self.z_loop):
            for param in self.ll_model.trainable_params():
                if param.grad is not None:
                    param.grad.set_data(mnp.zeros_like(param.grad))

            reg_decay = Tensor(reg_decay, ms.float32)
            self.ll_l2_reg = Tensor(self.ll_l2_reg, ms.float32)

            def compute_loss_z(ll_feed_dict, ul_model, ll_model):
                """
                Compute the total loss (loss_z) which includes:
                - The lower-level objective (loss_z_).
                - The L2 regularization term (loss_l2_z).

                :param ll_feed_dict: Lower-level feed dictionary.
                :param ul_model: Upper-level model.
                :param ll_model: Lower-level model.
                :returns: The total loss (loss_z).
                """
                # Compute the L2 regularization term
                loss_l2_z = (
                    self.ll_l2_reg / reg_decay * l2_reg(ll_model.trainable_params())
                )

                # Compute the lower-level objective
                loss_z_ = self.ll_objective(ll_feed_dict, ul_model, ll_model)

                # Combine losses
                loss_z = loss_z_ + loss_l2_z

                return loss_z

            # 计算梯度：loss_z
            grads = ops.GradOperation(get_by_list=True)(
                lambda: compute_loss_z(ll_feed_dict, self.ul_model, self.ll_model),
                self.ll_model.trainable_params(),
            )
            new_grads = grads()
            for param, grad in zip(self.ll_model.trainable_params(), new_grads):
                param.set_data(param - self.z_lr * grad)

        # Auxiliary model and its optimization
        auxiliary_model = copy.deepcopy(self.ll_model)
        auxiliary_opt = nn.SGD(
            auxiliary_model.trainable_params(), learning_rate=self.z_lr
        )

        loss_l2_z = (
            self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.trainable_params())
        )

        # 计算辅助模型的目标损失
        loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)

        # 合并总损失
        loss_z = loss_z_ + loss_l2_z

        # 清零梯度
        def compute_loss_y(
            ll_feed_dict, ul_feed_dict, ul_model, auxiliary_model, reg_decay
        ):
            """
            Compute the total loss (loss_y) for the auxiliary model update step.
            Includes:
            - Upper-level objective (loss_y_).
            - Regularization term (loss_l2_y).
            - Logarithmic loss term (loss_ln).

            :param ll_feed_dict: Lower-level feed dictionary.
            :param ul_feed_dict: Upper-level feed dictionary.
            :param ul_model: Upper-level model.
            :param auxiliary_model: Auxiliary model.
            :param reg_decay: Regularization decay factor.
            :returns: The total loss (loss_y).
            """
            # Compute individual terms
            loss_y_f_ = self.ll_objective(ll_feed_dict, ul_model, auxiliary_model)
            loss_y_ = self.ul_objective(ul_feed_dict, ul_model, auxiliary_model)
            loss_l2_y = (
                self.ul_l2_reg / reg_decay * l2_reg(auxiliary_model.trainable_params())
            )
            loss_ln = ops.log(loss_y_f_ + loss_z - loss_y_f_)  # Logarithmic term
            loss_ln = self.ul_ln_reg / reg_decay * loss_ln

            # Combine terms
            loss_y = loss_y_ - loss_ln + loss_l2_y
            return loss_y

        for y_idx in range(self.y_loop):
            # 清零梯度
            for param in auxiliary_model.trainable_params():
                if param.grad is not None:
                    param.grad.set_data(mnp.zeros_like(param.grad))
            # 计算梯度
            grads = ops.GradOperation(get_by_list=True)(
                lambda: compute_loss_y(
                    ll_feed_dict,
                    ul_feed_dict,
                    self.ul_model,
                    auxiliary_model,
                    reg_decay,
                ),
                auxiliary_model.trainable_params(),
            )()
            # print("auxiliary_model grads:", grads)

            # 使用优化器更新参数
            auxiliary_opt(grads)
        # 更新辅助模型
        # for param, grad in zip(auxiliary_model.trainable_params(), grads):
        #     param.set_data(param - self.z_lr * grad)

        # 更新上层模型
        def compute_loss_x(
            ll_feed_dict,
            ul_feed_dict,
            ul_model,
            ll_model,
            auxiliary_model,
            reg_decay,
            ul_ln_reg,
            ll_l2_reg,
        ):
            """
            Compute the total loss (loss_x) for the upper-level update.

            :param ll_feed_dict: Lower-level feed dictionary.
            :param ul_feed_dict: Upper-level feed dictionary.
            :param ul_model: Upper-level model.
            :param ll_model: Lower-level model.
            :param auxiliary_model: Auxiliary model.
            :param reg_decay: Regularization decay factor.
            :param ul_ln_reg: Logarithmic regularization factor.
            :param ll_l2_reg: Lower-level L2 regularization factor.
            :returns: The total loss (loss_x).
            """
            # Compute L2 regularization term
            loss_l2_z = ll_l2_reg / reg_decay * l2_reg(ll_model.trainable_params())

            # Compute lower-level objective
            loss_z_ = self.ll_objective(ll_feed_dict, ul_model, ll_model)
            loss_z = loss_z_ + loss_l2_z

            # Compute logarithmic regularization term
            loss_y_f_ = self.ll_objective(ll_feed_dict, ul_model, auxiliary_model)
            loss_ln = (
                ul_ln_reg / reg_decay * ops.log(loss_y_f_ + loss_z - loss_y_f_ + 1e-8)
            )  # Avoid log(0) error

            # Compute upper-level objective
            loss_x_ = self.ul_objective(ul_feed_dict, ul_model, auxiliary_model)
            loss_x = loss_x_ - loss_ln

            return loss_x

        # Use GradOperation to calculate gradients
        grad_fn = ops.GradOperation(get_by_list=True)(
            lambda: compute_loss_x(
                ll_feed_dict,
                ul_feed_dict,
                self.ul_model,
                self.ll_model,
                auxiliary_model,
                reg_decay,
                self.ul_ln_reg,
                self.ll_l2_reg,
            ),
            self.ul_var,
        )
        ul_grads = grad_fn()
        # print("UL_grads:", ul_grads)

        # 调用 compute_loss_x 计算损失值
        loss_x_value = compute_loss_x(
            ll_feed_dict,
            ul_feed_dict,
            self.ul_model,
            self.ll_model,
            auxiliary_model,
            reg_decay,
            self.ul_ln_reg,
            self.ll_l2_reg,
        )
        # print("Loss_x Value:", loss_x_value)
        # 使用优化器更新参数
        self.ul_opt(ul_grads)

        return loss_x_value.item()
