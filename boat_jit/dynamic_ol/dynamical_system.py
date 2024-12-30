import abc
from typing import List, Dict
from boat_jit.utils import DynamicalSystemRules, ResultStore
importlib = __import__("importlib")


class DynamicalSystem(object):
    def __init__(self, ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config) -> None:
        """
        Implements the abstract class for the lower-level optimization procedure.

        Parameters
        ----------
            :param ll_objective: The lower-level objective of the BLO problem.
            :type ll_objective: callable
            :param ll_model: The lower-level model of the BLO problem.
            :type ll_model: torch.nn.Module
            :param ul_model: The upper-level model of the BLO problem.
            :type ul_model: torch.nn.Module
            :param lower_loop: Number of iterations for lower-level optimization.
            :type lower_loop: int
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


class SequentialDS:
    """
    A dynamically created class for sequential hyper-gradient operations.
    """
    def __init__(self, ordered_instances: List[object], custom_order: List[str]):
        self.gradient_instances = ordered_instances
        self.custom_order = custom_order
        self.result_store = ResultStore()  # Use a dedicated result store

    def optimize(self, **kwargs) -> List[Dict]:
        """
        Compute gradients sequentially using the ordered instances.

        :param kwargs: Arguments required for gradient computations.
        :return: A list of dictionaries containing results for each gradient operator.
        """
        self.result_store.clear()  # Reset the result store
        intermediate_result = None

        for idx, gradient_instance in enumerate(self.gradient_instances):
            # Compute the gradient, passing the intermediate result as input
            result = gradient_instance.optimize(
                **(kwargs if idx == 0 else intermediate_result), next_operation=self.custom_order[idx + 1]
                if idx + 1 < len(self.custom_order) else None
            )
            # Store the result
            self.result_store.add(f"dynamic_results_{idx}", result)
            intermediate_result = result

        return self.result_store.get_results()


def makes_functional_dynamical_system(
        custom_order: List[str],
        **kwargs
) -> SequentialDS:
    """
    Dynamically create a SequentialHyperGradient object with ordered gradient operators.

    Parameters
    ----------
    custom_order : List[str]
        User-defined operator order.

    Returns
    -------
    SequentialHyperGradient
        An instance with ordered gradient operators and result management.
    """
    # Load the predefined gradient order
    gradient_order = DynamicalSystemRules.get_gradient_order()

    # Adjust custom order based on predefined gradient order
    adjusted_order = validate_and_adjust_order(custom_order, gradient_order)

    # Dynamically load classes
    gradient_classes = {}
    module = importlib.import_module("boat_jit.dynamic_ol")
    for op in custom_order:
        gradient_classes[op] = getattr(module, op)

    # Reorder classes according to adjusted order
    ordered_instances = [gradient_classes[op](**kwargs) for op in adjusted_order]

    # Return the enhanced sequential hyper-gradient class
    return SequentialDS(ordered_instances, custom_order)


def validate_and_adjust_order(custom_order: List[str], gradient_order: List[List[str]]) -> List[str]:
    """
    Validate and adjust the custom order to match the predefined gradient order.

    Parameters
    ----------
    custom_order : List[str]
        The user-provided order of gradient operators.
    gradient_order : List[List[str]]
        The predefined order of gradient operator groups.

    Returns
    -------
    List[str]
        Adjusted order of gradient operators following the predefined rules.
    """
    # Create a set of valid operators for quick lookup
    valid_operators = {op for group in gradient_order for op in group}

    # Filter out invalid operators
    custom_order = [op for op in custom_order if op in valid_operators]

    # Adjust order to follow gradient_order
    adjusted_order = []
    for group in gradient_order:
        for op in group:
            if op in custom_order:
                adjusted_order.append(op)

    return adjusted_order
