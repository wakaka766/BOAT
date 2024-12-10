import time
import importlib
from typing import Dict, Any, Callable
import torch
import copy
from torch import Tensor
from torch.optim import Optimizer
import higher


from boat.utils.op_utils import copy_parameter_from_list,average_grad
importlib = __import__("importlib")
ll_grads = importlib.import_module("boat.dynamic_ol")
ul_grads = importlib.import_module("boat.hyper_ol")
fo_gms = importlib.import_module("boat.fogm")
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
        self._fo_gm = config["fo_gm"]
        self._dynamic_op = config["dynamic_op"]
        self._hyper_op = config["hyper_op"]
        self._ll_loss = self._load_loss_function(loss_config["lower_level_loss"])
        self._ul_loss = self._load_loss_function(loss_config["upper_level_loss"])
        self._ll_model = config["lower_level_model"]
        self._ul_model = config["upper_level_model"]
        self._ll_var = list(config["lower_level_var"])
        self._ul_var = list(config["upper_level_var"])
        self.boat_configs = config
        self.boat_configs["gda_loss"] = self._load_loss_function(loss_config["GDA_loss"]) if 'GDA' in config["dynamic_op"] else None
        self._ll_solver = None
        self._ul_solver = None
        self._lower_opt = None
        self._upper_opt = None
        self._device = torch.device(config["device"])

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

    def build_ll_solver(self, lower_opt: Optimizer):
        """
        Configure the lower-level solver.

        Args:
            lower_opt (Optimizer): Optimizer for the lower-level model.
            solver_config (Dict[str, Any]): Configuration for the lower-level solver.
        """
        if self.boat_configs['fo_gm'] is None:
            assert (self.boat_configs[
                       'dynamic_op'] is not None) and (self.boat_configs['hyper_op'] is not None), "Set 'dynamic_op' and 'hyper_op' properly."
            sorted_ops = sorted([op.upper() for op in self._dynamic_op])
            dynamic_ol = "_".join(sorted_ops)
            self._lower_opt = lower_opt
            self.boat_configs['ll_opt'] =self._lower_opt
            self._lower_loop = self.boat_configs.get("lower_iters", 10)
            self.check_status()
            if 'DM' in self._dynamic_op:
                self.boat_configs["DM"]['auxiliary_v'] = [torch.zeros_like(param) for param in self._ll_var]
                self.boat_configs["DM"]['auxiliary_v_opt'] = torch.optim.SGD(self.boat_configs["DM"]['auxiliary_v'], lr=self.boat_configs["DM"]['auxiliary_v_lr'])
            if self._hyper_op == 'FD':
                assert self.boat_configs['lower_iters'] == 1, "Finite Differentiation method requires one gradient step to optimize task parameters."
                assert self.boat_configs['RGT']["truncate_iter"] == 0 and not ("PTT" in self.boat_configs["hyper_op"]), \
                    "One-stage method doesn't need trajectory truncation."
            self._ll_solver = getattr(
                ll_grads, "%s" % dynamic_ol
            )(ll_objective=self._ll_loss,
              ul_objective=self._ul_loss,
              ll_model=self._ll_model,
              ul_model=self._ul_model,
              lower_loop=self._lower_loop,
              solver_config=self.boat_configs)
        else:
            self._lower_opt = lower_opt
            self.boat_configs['ll_opt'] =self._lower_opt
            self._lower_loop = self.boat_configs.get("lower_iters", 10)
            self.fogm_solver = getattr(
                fo_gms, "%s" % self.boat_configs['fo_gm']
            )(ll_objective=self._ll_loss,
              ul_objective=self._ul_loss,
              ll_model=self._ll_model,
              ul_model=self._ul_model,
              lower_loop=self._lower_loop,
              ll_opt=self._lower_opt,
              ll_var=self._ll_var,
              ul_var=self._ul_var,
              solver_config=self.boat_configs)
        return self

    def build_ul_solver(self,upper_opt: Optimizer):
        """
        Configure the upper-level solver.

        Args:
            upper_opt (Optimizer): Optimizer for the upper-level model.
            solver_config (Dict[str, Any]): Configuration for the upper-level solver.
        """
        self._upper_opt = upper_opt
        if self.boat_configs['fo_gm'] is None:
            assert self.boat_configs['hyper_op'] is not None, "Choose FOGM based methods from ['VSM','VFM','MESM'] or set 'dynamic_ol' and 'hyper_ol' properly."
            sorted_ops = sorted([op.upper() for op in self._hyper_op])
            hyper_op = "_".join(sorted_ops)
            if 'DM' in self._dynamic_op:
                # if not hasattr(self._ll_solver, 'ul_opt'):
                    # 如果没有，则为 _ll_solver 添加该属性
                setattr(self._ll_solver, 'ul_opt', upper_opt)  # 设置 new_attribute 属性
                setattr(self._ll_solver, 'ul_lr', upper_opt.defaults['lr'])
            # self._total_iters = self.boat_configs.get("total_iters", 60000)
            if "DI" in self.boat_configs["dynamic_op"]:
                self._lower_init_opt = copy.deepcopy(self._lower_opt)
                self._lower_init_opt.param_groups[0]['params'] = self._lower_opt.param_groups[0]['params']
                self._lower_init_opt.param_groups[0]['lr'] = self.boat_configs["DI"]["lr"]
            # if self._hyper_op == "RAD":
            #     # Placeholder for specific solver initialization
            #     pass
            # elif self._hyper_op == "LS":
            #     # Placeholder for specific solver initialization
            #     pass
            # else:
            #     raise ValueError(f"Unsupported upper-level method: {self._hyper_op}")

            self._ul_solver = getattr(
                ul_grads, "%s" % hyper_op
            )(ul_objective=self._ul_loss,
              ll_objective=self._ll_loss,
              ll_model=self._ll_model,
              ul_model=self._ul_model,
              ll_var = self._ll_var,
              ul_var = self._ul_var,
              solver_config=self.boat_configs)
        else:
            assert self.boat_configs['fo_gm'] is not None, "Choose FOGM based methods from ['VSM','VFM','MESM'] or set 'dynamic_ol' and 'hyper_ol' properly."

        return self

    def run_iter(self, ll_feed_dict: Dict[str, Tensor], ul_feed_dict: Dict[str, Tensor],
                 current_iter: int) -> tuple:
        """
        Run a single iteration of the optimization process with flexible multi-modality input.

        Args:
            ll_feed_dict (Dict[str, Tensor]): Dictionary containing all training data.
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

        if self.boat_configs['fo_gm'] is not None:

            start_time = time.time()
            loss = self.fogm_solver.optimize(ll_feed_dict, ul_feed_dict, current_iter)
            run_time = time.time() - start_time
        else:
            run_time = 0
            if self.boat_configs['accumulate_grad']:
                assert "IAD" in self.boat_configs['hyper_op'], "When using 'accumulate_grad', only 'IAD' based methods are supported."
                for batch_ll_feed_dict,batch_ul_feed_dict in zip(ll_feed_dict,ul_feed_dict):
                    with higher.innerloop_ctx(self._ll_model, self._lower_opt,
                                              copy_initial_weights=False) as (auxiliary_model, auxiliary_opt):
                        forward_time = time.time()
                        max_loss_iter = self._ll_solver.optimize(batch_ll_feed_dict, batch_ul_feed_dict, auxiliary_model, auxiliary_opt,current_iter)
                        forward_time = time.time() - forward_time
                        backward_time = time.time()
                        loss = self._ul_solver.compute_gradients(batch_ll_feed_dict, batch_ul_feed_dict, auxiliary_model, max_loss_iter)
                        backward_time = time.time() - backward_time
                    run_time += forward_time + backward_time
                if "DI" in self.boat_configs['dynamic_op']:
                    self._lower_init_opt.step()
                    self._lower_init_opt.zero_grad()
                average_grad(self._ul_model,len(ll_feed_dict))


            else:
                with higher.innerloop_ctx(self._ll_model, self._lower_opt,
                                          copy_initial_weights=True) as (auxiliary_model, auxiliary_opt):
                    forward_time = time.time()
                    max_loss_iter = self._ll_solver.optimize(ll_feed_dict, ul_feed_dict, auxiliary_model, auxiliary_opt, current_iter)
                    forward_time = time.time() - forward_time
                    backward_time = time.time()
                    if "DM" not in self._dynamic_op:
                        loss = self._ul_solver.compute_gradients(ll_feed_dict, ul_feed_dict, auxiliary_model, max_loss_iter)
                    else:
                        loss = self._ul_loss(ul_feed_dict,self._ul_model, auxiliary_model)
                    backward_time = time.time() - backward_time
                    if ("DM" not in self._dynamic_op) and ("DI" not in self._dynamic_op) and ("IAD" not in self._hyper_op):
                        copy_parameter_from_list(self._ll_model, list(auxiliary_model.parameters(time=max_loss_iter)))

                # update adapt_model parameters
                if "DI" in self.boat_configs['dynamic_op']:
                    self._lower_init_opt.step()
                    self._lower_init_opt.zero_grad()
                run_time = forward_time + backward_time
            if not self.boat_configs['return_grad']:
                self._upper_opt.step()
                self._upper_opt.zero_grad()
            else:
                return [var.grad for var in list(self._ul_var)], run_time

        return loss, run_time

    def check_status(self):
        # assert self.boat_configs['RGT']["truncate_iter"] == 0 or not ("PTT" in self.boat_configs["hyper_op"]), \
        #     "Only one of the PTT and RGT methods could be chosen."
        if "RGT" in self.boat_configs["dynamic_op"]:
            assert self.boat_configs['RGT']["truncate_iter"] > 0, \
                "When RGT is chosen, set the 'truncate_iter' properly ."
        assert (("DI" in self._dynamic_op )^ ("IAD" in self._hyper_op)) or (("DI" not in self._dynamic_op) and ("IAD" not in self._hyper_op)), \
            "Only one of the PTT and RGT methods could be chosen."
        assert (
                0.0 <= self.boat_configs['GDA']["alpha_init"] <= 1.0
        ), "Parameter 'alpha' used in method BDA should be in the interval (0,1)."
        if self.boat_configs['GDA']["alpha_init"] > 0.0:
            assert (0.0 < self.boat_configs['GDA']["alpha_decay"] <= 1.0), \
                "Parameter 'alpha_decay' used in method BDA should be in the interval (0,1)."
        assert (self.boat_configs['RGT']["truncate_iter"] < self.boat_configs[
            'lower_iters']), "The value of 'truncate_iter' shouldn't be greater than 'lower_loop'."