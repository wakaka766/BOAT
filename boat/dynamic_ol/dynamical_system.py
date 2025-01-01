import abc
from typing import Dict
importlib = __import__("importlib")


class DynamicalSystem(object):
    def __init__(
        self, ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
    ) -> None:
        """
        Abstract class for defining lower-level optimization procedures in Bilevel Optimization (BLO).

        Parameters
        ----------
        :param ll_objective: The lower-level objective function of the BLO problem.
        :type ll_objective: Callable
        :param ul_objective: The upper-level objective function of the BLO problem.
        :type ul_objective: Callable
        :param ll_model: The lower-level model of the BLO problem.
        :type ll_model: torch.nn.Module
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: torch.nn.Module
        :param lower_loop: The number of iterations for lower-level optimization.
        :type lower_loop: int
        :param solver_config: A dictionary containing solver configurations. It includes details about optimization algorithms,
            hyperparameter settings, and additional configurations required for solving the BLO problem.
        :type solver_config: Dict[str, Any]
        """
        self.ll_objective = ll_objective
        self.ul_objective = ul_objective
        self.lower_loop = lower_loop
        self.ul_model = ul_model
        self.ll_model = ll_model
        self.solver_config = solver_config

    @abc.abstractmethod
    def optimize(self, **kwargs):
        pass

