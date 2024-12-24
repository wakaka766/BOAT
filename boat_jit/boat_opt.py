import time
import copy
from typing import Dict, Any, Callable
from boat_jit.utils.op_utils import copy_parameter_from_list, average_grad, manual_update
try:
    import jittor as jit
    from jittor.optim import Optimizer
    import boat_jit.higher_jit as higher
except ImportError as e:
    missing_module = str(e).split()[-1]
    print(f"Error: The required module '{missing_module}' is not installed.")
    print("Please run the following command to install all required dependencies:")
    print("pip install -r requirements.txt")
    raise

importlib = __import__("importlib")
ll_grads = importlib.import_module("boat_jit.dynamic_ol")
ul_grads = importlib.import_module("boat_jit.hyper_ol")
fo_gms = importlib.import_module("boat_jit.fogm")


def _load_loss_function(loss_config: Dict[str, Any]) -> Callable:
    """
    Dynamically load a loss function from the provided configuration.

    :param loss_config: Dictionary with keys:
        - "function": Path to the loss function (e.g., "module.path.to_function").
        - "params": Parameters to be passed to the loss function.
    :type loss_config: Dict[str, Any]

    :returns: Loaded loss function ready for use.
    :rtype: Callable
    """

    module_name, func_name = loss_config["function"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    # Return a wrapper function that can accept both positional and keyword arguments
    return lambda *args, **kwargs: func(*args, **{**loss_config.get("params", {}), **kwargs})


class Problem:
    """
    Enhanced bi-level optimization problem class supporting flexible loss functions and operation configurations.
    """

    def __init__(self, config: Dict[str, Any], loss_config: Dict[str, Any]):
        """
        Initialize the Problem instance.

        :param config: Configuration dictionary for the optimization setup.
            - "fo_gm": First Order Gradient based Method (optional), e.g., ["VSM"], ["VFM"], ["MESM"].
            - "dynamic_op": List of dynamic operations (optional), e.g., ["NGD"], ["NGD", "GDA"], ["NGD", "GDA", "DI"].
            - "hyper_op": Hyper-optimization method (optional), e.g., ["RAD"], ["RAD", "PTT"], ["IAD", "NS", "PTT"].
            - "lower_level_loss": Configuration for the lower-level loss function based on the json file configuration.
            - "upper_level_loss": Configuration for the upper-level loss function based on the json file configuration.
            - "lower_level_model": The lower-level model to be optimized.
            - "upper_level_model": The upper-level model to be optimized.
            - "lower_level_var": Variables in the lower-level model.
            - "upper_level_var": Variables in the upper-level model.
        :type config: Dict[str, Any]

        :param loss_config: Loss function configuration dictionary.
            - "lower_level_loss": Configuration for the lower-level loss function.
            - "upper_level_loss": Configuration for the upper-level loss function.
            - "GDA_loss": Configuration for GDA loss function (optional).
        :type loss_config: Dict[str, Any]

        :returns: None
        """
        self._fo_gm = config["fo_gm"]
        self._dynamic_op = config["dynamic_op"]
        self._hyper_op = config["hyper_op"]
        self._ll_model = config["lower_level_model"]
        self._ul_model = config["upper_level_model"]
        self._ll_var = list(config["lower_level_var"])
        self._ul_var = list(config["upper_level_var"])
        self.boat_configs = config
        self.boat_configs["gda_loss"] = _load_loss_function(loss_config["gda_loss"]) \
            if 'GDA' in config["dynamic_op"] else None
        self._ll_loss = _load_loss_function(loss_config["lower_level_loss"])
        self._ul_loss = _load_loss_function(loss_config["upper_level_loss"])
        self._ll_solver = None
        self._ul_solver = None
        self._lower_opt = None
        self._upper_opt = None
        self._lower_init_opt = None
        self._fo_gm_solver = None
        self._lower_loop = None
        self._log_results_dict = {}

    def build_ll_solver(self, lower_opt: Optimizer):
        """
        Configure the lower-level solver.

        :param lower_opt: The optimizer to use for the lower-level variables initialized (defined in the 'config["lower_level_var"]').
        :type lower_opt: Optimizer

        :returns: None
        """
        if self.boat_configs['fo_gm'] is None:
            assert (self.boat_configs[
                        'dynamic_op'] is not None) and (
                           self.boat_configs['hyper_op'] is not None), "Set 'dynamic_op' and 'hyper_op' properly."
            sorted_ops = sorted([op.upper() for op in self._dynamic_op])
            dynamic_ol = "_".join(sorted_ops)
            self._lower_opt = lower_opt
            self.boat_configs['ll_opt'] = self._lower_opt
            self._lower_loop = self.boat_configs.get("lower_iters", 10)
            self.check_status()
            if 'DM' in self._dynamic_op:
                self.boat_configs["DM"]['auxiliary_v'] = [jit.zeros_like(param) for param in self._ll_var]
                self.boat_configs["DM"]['auxiliary_v_opt'] = jit.nn.SGD(self.boat_configs["DM"]['auxiliary_v'],
                                                                    lr=self.boat_configs["DM"]['auxiliary_v_lr'])
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
            self.boat_configs['ll_opt'] = self._lower_opt
            self._lower_loop = self.boat_configs.get("lower_iters", 10)
            self._fo_gm_solver = getattr(
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

    def build_ul_solver(self, upper_opt: Optimizer):
        """
        Configure the lower-level solver.

        :param upper_opt: The optimizer to use for the lower-level variables initialized (defined in the 'config["lower_level_var"]').
        :type upper_opt: Optimizer

        :returns: None
        """
        self._upper_opt = upper_opt
        if self.boat_configs['fo_gm'] is None:
            assert self.boat_configs['hyper_op'] is not None, \
                "Choose FOGM based methods from ['VSM'],['VFM'],['MESM'] or set 'dynamic_ol' and 'hyper_ol' properly."
            sorted_ops = sorted([op.upper() for op in self._hyper_op])
            hyper_op = "_".join(sorted_ops)
            if "DM" in self._dynamic_op:
                setattr(self._ll_solver, 'ul_opt', upper_opt)  # 设置 new_attribute 属性
                setattr(self._ll_solver, 'ul_lr', upper_opt.defaults['lr'])
            if "DI" in self.boat_configs["dynamic_op"]:
                self._lower_init_opt = copy.deepcopy(self._lower_opt)
                for _ in range(len(self._lower_init_opt.param_groups)):
                    self._lower_init_opt.param_groups[_]['params'] = self._lower_opt.param_groups[_]['params']
                    self._lower_init_opt.param_groups[_]['lr'] = self.boat_configs["DI"]["lr"]
            self._ul_solver = getattr(
                ul_grads, "%s" % hyper_op
            )(ul_objective=self._ul_loss,
              ll_objective=self._ll_loss,
              ll_model=self._ll_model,
              ul_model=self._ul_model,
              ll_var=self._ll_var,
              ul_var=self._ul_var,
              solver_config=self.boat_configs)
        else:
            assert self.boat_configs['fo_gm'] is not None, \
                "Choose FOGM based methods from ['VSM','VFM','MESM'] or set 'dynamic_ol' and 'hyper_ol' properly."

        return self

    def run_iter(self, ll_feed_dict: Dict[str, jit.Var], ul_feed_dict: Dict[str, jit.Var], current_iter: int) -> tuple:
        """
           Run a single iteration of the bi-level optimization process.

           :param ll_feed_dict: Dictionary containing the real-time data and parameters fed for the construction of the lower-level (LL) objective.
               Example:
                   {
                       "image": train_images,
                       "text": train_texts,
                       "target": train_labels  # Optional
                   }
           :type ll_feed_dict: Dict[str, Tensor]

           :param ul_feed_dict: Dictionary containing the real-time data and parameters fed for the construction of the upper-level (UL) objective.
               Example:
                   {
                       "image": val_images,
                       "text": val_texts,
                       "target": val_labels  # Optional
                   }
           :type ul_feed_dict: Dict[str, Tensor]

           :param current_iter: The current iteration number.
           :type current_iter: int

           :notes:
               - When `accumulate_grad` is set to True, you need to pack the data of each batch based on the format above.
               - In that case, pass `ll_feed_dict` and `ul_feed_dict` as lists of dictionaries, i.e., `[Dict[str, Tensor]]`.

           :returns: A tuple containing:
               - loss (float): The loss value for the current iteration.
               - run_time (float): The total time taken for the iteration.
           :rtype: tuple
           """
        self._log_results_dict['upper_loss'] = []
        if self.boat_configs['fo_gm'] is not None:
            start_time = time.perf_counter()
            self._log_results_dict['upper_loss'].append(
                self._fo_gm_solver.optimize(ll_feed_dict, ul_feed_dict, current_iter))
            run_time = time.perf_counter() - start_time
        else:
            run_time = 0
            if self.boat_configs['accumulate_grad']:
                for batch_ll_feed_dict, batch_ul_feed_dict in zip(ll_feed_dict, ul_feed_dict):
                    with higher.innerloop_ctx(self._ll_model, self._lower_opt,
                                              copy_initial_weights=False) as (auxiliary_model, auxiliary_opt):
                        forward_time = time.perf_counter()
                        max_loss_iter = self._ll_solver.optimize(batch_ll_feed_dict, batch_ul_feed_dict,
                                                                 auxiliary_model, auxiliary_opt, current_iter)
                        forward_time = time.perf_counter() - forward_time
                        backward_time = time.perf_counter()
                        self._log_results_dict['upper_loss'].append(
                            self._ul_solver.compute_gradients(batch_ll_feed_dict, batch_ul_feed_dict,
                                                              auxiliary_model, max_loss_iter))
                        backward_time = time.perf_counter() - backward_time
                    run_time += forward_time + backward_time
                average_grad(self._ul_model, len(ll_feed_dict))
            else:
                with higher.innerloop_ctx(self._ll_model, self._lower_opt,
                                          copy_initial_weights=True) as (auxiliary_model, auxiliary_opt):
                    forward_time = time.perf_counter()
                    max_loss_iter = self._ll_solver.optimize(ll_feed_dict, ul_feed_dict, auxiliary_model, auxiliary_opt,
                                                             current_iter)
                    forward_time = time.perf_counter() - forward_time
                    backward_time = time.time()
                    if "DM" not in self._dynamic_op:
                        self._log_results_dict['upper_loss'].append(
                            self._ul_solver.compute_gradients(ll_feed_dict, ul_feed_dict, auxiliary_model,
                                                              max_loss_iter))
                    else:
                        self._log_results_dict['upper_loss'].append(
                            self._ul_loss(ul_feed_dict, self._ul_model, auxiliary_model))
                    backward_time = time.perf_counter() - backward_time
                    if ("DM" not in self._dynamic_op) and ("DI" not in self._dynamic_op) and (
                            "IAD" not in self._hyper_op):
                        copy_parameter_from_list(self._ll_model, list(auxiliary_model.parameters(time=max_loss_iter)))
                # update the dynamic initialization of lower-level variables
                if "DI" in self.boat_configs['dynamic_op']:
                    # self._lower_init_opt.step()
                    # self._lower_init_opt.zero_grad()
                    manual_update(self._lower_init_opt,self._lower_opt.param_groups[0]['params'])
                run_time = forward_time + backward_time
        if not self.boat_configs['return_grad']:
            # self._upper_opt.step()
            # self._upper_opt.zero_grad()
            manual_update(self._upper_opt,self._ul_var)
        else:
            return [var._custom_grad for var in list(self._ul_var)], run_time

        return self._log_results_dict['upper_loss'], run_time

    def check_status(self):

        if "DM" in self.boat_configs["dynamic_op"]:
            assert (self.boat_configs["hyper_op"] == ["RAD"]) or (self.boat_configs["hyper_op"] == ["CG"]), \
                "When 'DM' is chosen, set the 'truncate_iter' properly."
        if "RGT" in self.boat_configs["hyper_op"]:
            assert self.boat_configs['RGT']["truncate_iter"] > 0, \
                "When 'RGT' is chosen, set the 'truncate_iter' properly ."
        if self.boat_configs['accumulate_grad']:
            assert "IAD" in self.boat_configs[
                'hyper_op'], "When using 'accumulate_grad', only 'IAD' based methods are supported."
        if self.boat_configs['GDA']["alpha_init"] > 0.0:
            assert (0.0 < self.boat_configs['GDA']["alpha_decay"] <= 1.0), \
                "Parameter 'alpha_decay' used in method BDA should be in the interval (0,1)."
        if 'FD' in self._hyper_op:
            assert self.boat_configs['RGT']["truncate_iter"] == 0, \
                "One-stage method doesn't need trajectory truncation."

        def check_model_structure(base_model, meta_model):
            for param1, param2 in zip(base_model.parameters(), meta_model.parameters()):
                if (param1.shape != param2.shape) or (param1.dtype != param2.dtype) or (param1.device != param2.device):
                    return False
            return True

        if "IAD" in self._hyper_op:
            assert (check_model_structure(self._ll_model, self._ul_model)), \
                ("With IAD or FOA operation, 'upper_level_model' and 'lower_level_model' have the same structure, "
                 "and 'lower_level_var' and 'upper_level_var' are the same group of variables.")
        assert (("DI" in self._dynamic_op) ^ ("IAD" in self._hyper_op)) or (
                ("DI" not in self._dynamic_op) and ("IAD" not in self._hyper_op)), \
            "Only one of the 'PTT' and 'RGT' methods could be chosen."
        assert (
                0.0 <= self.boat_configs['GDA']["alpha_init"] <= 1.0
        ), "Parameter 'alpha' used in method BDA should be in the interval (0,1)."
        assert (self.boat_configs['RGT']["truncate_iter"] < self.boat_configs[
            'lower_iters']), "The value of 'truncate_iter' shouldn't be greater than 'lower_loop'."


