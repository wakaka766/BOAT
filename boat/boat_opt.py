import time
import importlib
from typing import Dict, Any, Callable
import torch
import copy
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from model.backbone import initialize
import higher

# 导入辅助模块
from boat.hyper_ol.iad import IAD
from boat.utils.op_utils import copy_parameter_from_list
importlib = __import__("importlib")
ll_grads = importlib.import_module("boat.dynamic_ol")
ul_grads = importlib.import_module("boat.hyper_ol")

class Problem:
    """
    Enhanced bi-level optimization problem class supporting flexible loss functions and retaining original logic.
    """

    def __init__(self, config: Dict[str, Any], loss_config: Dict[str, Any]):
        """
        Initialize the Problem instance.

        Args:
            config (Dict[str, Any]): Configuration dictionary including:
                - "method": Optimization method ("Feature" or "Initial").
                - "ll_method": Lower-level optimization method.
                - "ul_method": Upper-level optimization method.
                - "ll_loss": Lower-level loss configuration (function path and params).
                - "ul_loss": Upper-level loss configuration (function path and params).
                - "ll_model": Lower-level model.
                - "ul_model": Upper-level model.
                - "total_iters": Total iterations.
                - Additional configurations for solvers and optimizers.
        """
        self._method = config["method"]
        self._dynamic_op = config["ll_method"]
        self._hyper_op = config["ul_method"]
        self._ll_loss = self._load_loss_function(loss_config["ll_loss"])
        self._ul_loss = self._load_loss_function(loss_config["ul_loss"])
        self._ll_model = config["ll_model"]
        self._ul_model = config["ul_model"]
        # self._total_iters = config.get("total_iters", 60000)
        self.boat_configs = config
        # Lower and upper solvers will be built later
        self._ll_solver = None
        self._ul_solver = None
        self._lower_opt = None
        self._upper_opt = None

    def _load_loss_function(self, loss_config: Dict[str, Any]) -> Callable:
        """
        Dynamically load a loss function from the provided configuration.

        Args:
            loss_config (Dict[str, Any]): Dictionary with keys:
                - "function": Path to the loss function (e.g., "module.path.to_function").
                - "params": Parameters to be passed to the loss function.

        Returns:
            Callable: Loaded loss function ready for use.
        """
        module_name, func_name = loss_config["function"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        # Return a wrapper function that can accept both positional and keyword arguments
        return lambda *args, **kwargs: func(*args, **{**loss_config.get("params", {}), **kwargs})



    def build_ll_solver(self,lower_model, lower_opt: Optimizer):
        """
        Configure the lower-level solver.

        Args:
            lower_opt (Optimizer): Optimizer for the lower-level model.
            solver_config (Dict[str, Any]): Configuration for the lower-level solver.
        """
        self._lower_opt = lower_opt
        self._ll_model = lower_model
        self.boat_configs['ll_opt'] =self._lower_opt
        self._lower_loop = self.boat_configs.get("lower_iters", 10)
        if self._dynamic_op == 'Dynamic':
            if self._hyper_op == 'FD':
                assert self.boat_configs['lower_iters'] == 1, "Finite Differentiation method requires one gradient step to optimize task parameters."
                assert self.boat_configs['RGT']["truncate_iter"] == 0 and not self.boat_configs['PTT']["truncate_max_loss_iter"] == 0, \
                    "One-stage method doesn't need trajectory truncation."
            assert self.boat_configs['RGT']["truncate_iter"] == 0 or not self.boat_configs['PTT']["truncate_max_loss_iter"], \
                "Only one of the IAPTT-GM and TRAD methods could be chosen."
            assert (
                    0.0 <= self.boat_configs['GDA']["alpha_init"] <= 1.0
            ), "Parameter 'alpha' used in method BDA should be in the interval (0,1)."
            if self.boat_configs['GDA']["alpha_init"] > 0.0:
                assert (0.0 < self.boat_configs['GDA']["alpha_decay"] <= 1.0), \
                    "Parameter 'alpha_decay' used in method BDA should be in the interval (0,1)."
            assert (self.boat_configs['RGT']["truncate_iter"] < self.boat_configs['lower_iters']), "The value of 'truncate_iter' shouldn't be greater than 'lower_loop'."
        # if self._dynamic_op == "Dynamic":
        #     # Placeholder for specific solver initialization
        #     pass
        # elif self._dynamic_op == "Implicit":
        #     # Placeholder for specific solver initialization
        #     pass
        # else:
        #     raise ValueError(f"Unsupported lower-level method: {self._dynamic_op}")
        self._ll_solver = getattr(
            ll_grads, "%s" % self._dynamic_op
        )(ll_objective=self._ll_loss,
          ul_objective=self._ul_loss,
          ll_model=self._ll_model,
          ul_model=self._ul_model,
          lower_loop=self._lower_loop,
          solver_config=self.boat_configs)


    def build_ul_solver(self,upper_model, upper_opt: Optimizer):
        """
        Configure the upper-level solver.

        Args:
            upper_opt (Optimizer): Optimizer for the upper-level model.
            solver_config (Dict[str, Any]): Configuration for the upper-level solver.
        """
        self._upper_opt = upper_opt
        self._ul_model = upper_model
        self._total_iters = self.boat_configs.get("total_iters", 60000)
        if self.boat_configs['update_ll_model_init']:
            assert self._dynamic_op == "Dynamic", \
                "Choose 'Dynamic' as ll method if you want to use initialization auxiliary."
            self._lower_init_opt = copy.deepcopy(self._lower_opt)
            self._lower_init_opt.param_groups[0]['lr'] = self._upper_opt.param_groups[0]['lr']
        # if self._hyper_op == "RAD":
        #     # Placeholder for specific solver initialization
        #     pass
        # elif self._hyper_op == "LS":
        #     # Placeholder for specific solver initialization
        #     pass
        # else:
        #     raise ValueError(f"Unsupported upper-level method: {self._hyper_op}")

        self._ul_solver = getattr(
            ul_grads, "%s" % self._hyper_op
        )(ul_objective=self._ul_loss,
          ll_model=self._ll_model,
          ul_model=self._ul_model,
          solver_config=self.boat_configs)

        return self

    def run_iter(self, train_data: Dict[str, Tensor], val_data: Dict[str, Tensor],
                 current_iter: int) -> tuple:
        """
        Run a single iteration of the optimization process with flexible multi-modality input.

        Args:
            train_data (Dict[str, Tensor]): Dictionary containing all training data.
                Example:
                    {
                        "image": train_images,
                        "text": train_texts,
                        "target": train_labels  # Optional
                    }
            val_data (Dict[str, Tensor]): Dictionary containing all validation data.
                Example:
                    {
                        "image": val_images,
                        "text": val_texts,
                        "target": val_labels  # Optional
                    }
            current_iter (int): Current iteration number.

        Returns:
            Tuple[float, float, float]: Tuple containing loss, forward time, and backward time.
        """

        kwargs_lower = {}
        kwargs_upper = {}
        losses = 0.0
        forward_time = 0.0
        backward_time = 0.0
        val_acc = 0.0
        # Prepare parameter dictionary for lower-level loss
        ll_params = {
            "data": train_data,  # Pass the entire training data dictionary
            "models": {"ll_model": self._ll_model, "ul_model": self._ul_model},
            "iter": current_iter
        }

        if self._method == "Initial":
            loss = self._meta_problem_solver.optimize(train_data, val_data)
            losses += loss.item()

        else:
            if self._dynamic_op == "BVFIM":
                forward_time = time.time()
                reg_decay = float(self.boat_configs['VSM']['reg_decay']) * current_iter + 1
                auxiliary_model = copy.deepcopy(self._ll_model)
                auxiliary_opt = torch.optim.SGD(auxiliary_model.parameters(), lr=0.01)
                out, auxiliary_model = self._ll_solver.optimize(train_data, auxiliary_model, auxiliary_opt,
                                                                val_data, reg_decay)
                # val_acc += accuary(out, val_y) / 4
                forward_time = time.time() - forward_time
                backward_time = time.time()
                loss = self._ul_solver.compute_gradients(val_data, auxiliary_model, train_data, reg_decay)
                backward_time = time.time() - backward_time
                initialize(self._ll_model)
            else:

                # Lower-level optimization
                with higher.innerloop_ctx(self._ll_model, self._lower_opt,
                                          copy_initial_weights=False) as (auxiliary_model, auxiliary_opt):
                    forward_time = time.time()
                    pmax = self._ll_solver.optimize(train_data,val_data, auxiliary_model, auxiliary_opt)
                    forward_time = time.time() - forward_time

                    # UL problem optimizing
                    backward_time = time.time()
                    if self.boat_configs["PTT"]['truncate_max_loss_iter']:
                        self.boat_configs['max_loss_iter'] = pmax

                    loss = self._ul_solver.compute_gradients(val_data, auxiliary_model, self.boat_configs)
                    backward_time = time.time() - backward_time
                copy_parameter_from_list(self._ll_model, auxiliary_model.parameters(time=-1))
            losses += loss.item()

        batch_size = 1
        # update adapt_model parameters
        if self._method == "Initial":
            if batch_size > 1:
                for x in self._meta_model.parameters():
                    x.grad = x.grad / batch_size
            self._meta_opt.step()
            self._meta_opt.zero_grad()
        else:
            if self._dynamic_op == "Dynamic":
                if self.boat_configs['update_ll_model_init']:
                    for x in self._ll_model.parameters():
                        x.grad = x.grad / batch_size
                    self._lower_init_opt.step()
                    self._lower_init_opt.zero_grad()

            self._upper_opt.step()
            self._upper_opt.zero_grad()

        return losses / batch_size, forward_time, backward_time

