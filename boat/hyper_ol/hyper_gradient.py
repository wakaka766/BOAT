import torch
from abc import abstractmethod


class HyperGradient(object):
    def __init__(self, ul_objective, ul_model, ll_model,ll_var,ul_var):
        self.ul_objective = ul_objective
        self.ul_model = ul_model
        self.ll_model = ll_model
        self.ll_var = ll_var
        self.ul_var = ul_var

    # _ERROR_HYPER_DETACHED = """
    # `The outer parameter` is detached from this optimization dynamics.
    # """
    #
    @abstractmethod
    def compute_gradients(self, **kwargs):
        r"""
        Should return a view or tuple of views (in correct shape) of the adapt_model's parameters.
        Implementation depends on specific adapt_model as adapt_model parameters may differ per adapt_model.

        Returns
        -------
        ndarray or tuple
            View or tuple of views of the adapt_model's parameters.
        """
        raise NotImplementedError("You should implement this!")

    """
        # Function overridden by specific methods.
        #
        # :param boml_inner_grad: inner_grad object resulting from the inner objective optimization.
        # :param outer_objective: A loss function for the outer parameters
        # :param meta_param: Optional list of outer parameters to consider. If not provided will get all variables in the
        #                     hyperparameter collection in the current scope.
        #
        # :return: list of outer parameters involved in the computation
        # """

