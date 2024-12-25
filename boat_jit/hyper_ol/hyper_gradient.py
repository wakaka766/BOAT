from abc import abstractmethod


class HyperGradient(object):
    def __init__(self, ul_objective, ul_model, ll_model, ll_var, ul_var):
        self.ul_objective = ul_objective
        self.ul_model = ul_model
        self.ll_model = ll_model
        self.ll_var = ll_var
        self.ul_var = ul_var

    @abstractmethod
    def compute_gradients(self, **kwargs):
        """
        Compute the hyper-gradients of upper-level variables.
        """
        raise NotImplementedError("You should implement this!")
