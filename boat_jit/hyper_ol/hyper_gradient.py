from abc import abstractmethod
from typing import List, Dict
from boat_jit.utils import HyperGradientRules
importlib = __import__("importlib")

class HyperGradient(object):
    def __init__(self, ll_objective, ul_objective, ul_model, ll_model, ll_var, ul_var, solver_config):
        self.ll_objective = ll_objective
        self.ul_objective = ul_objective
        self.ul_model = ul_model
        self.ll_model = ll_model
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.solver_configs = solver_config

    @abstractmethod
    def compute_gradients(self, **kwargs):
        """
        Compute the hyper-gradients of upper-level variables.
        """
        raise NotImplementedError("You should implement this!")


class ResultStore:
    """
    A simple class to store and manage intermediate results of hyper-gradient computation.
    """
    def __init__(self):
        self.results = []

    def add(self, name: str, result: Dict):
        """
        Add a result to the store.

        :param name: The name of the result (e.g., 'gradient_operator_results_0').
        :type name: str
        :param result: The result dictionary to store.
        :type result: Dict
        """
        self.results.append({name: result})

    def clear(self):
        """Clear all stored results."""
        self.results = []

    def get_results(self) -> List[Dict]:
        """Retrieve all stored results."""
        return self.results


class SequentialHyperGradient:
    """
    A dynamically created class for sequential hyper-gradient operations.
    """
    def __init__(self, ordered_instances: List[object], custom_order: List[str]):
        self.gradient_instances = ordered_instances
        self.custom_order = custom_order
        self.result_store = ResultStore()  # Use a dedicated result store

    def compute_gradients(self, **kwargs) -> List[Dict]:
        """
        Compute gradients sequentially using the ordered instances.

        :param kwargs: Arguments required for gradient computations.
        :return: A list of dictionaries containing results for each gradient operator.
        """
        self.result_store.clear()  # Reset the result store
        intermediate_result = None

        for idx, gradient_instance in enumerate(self.gradient_instances):
            # Compute the gradient, passing the intermediate result as input
            result = gradient_instance.compute_gradients(
                **(kwargs if idx == 0 else intermediate_result),next_operation=self.custom_order[idx + 1]
                if idx + 1 < len(self.custom_order) else None
            )
            # Store the result
            self.result_store.add(f"gradient_operator_results_{idx}", result)
            intermediate_result = result

        return self.result_store.get_results()


def makes_functional_hyper_operation(
        custom_order: List[str],
        **kwargs
) -> SequentialHyperGradient:
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
    gradient_order = HyperGradientRules.get_gradient_order()

    # Adjust custom order based on predefined gradient order
    adjusted_order = validate_and_adjust_order(custom_order, gradient_order)

    # Dynamically load classes
    gradient_classes = {}
    module = importlib.import_module("boat_jit.hyper_ol")
    for op in custom_order:
        gradient_classes[op] = getattr(module, op)

    # Reorder classes according to adjusted order
    ordered_instances = [gradient_classes[op](**kwargs) for op in adjusted_order]

    # Return the enhanced sequential hyper-gradient class
    return SequentialHyperGradient(ordered_instances, custom_order)


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
