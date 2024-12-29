import jittor as jit
from typing import List, Callable


class HyperGradientRules:
    """
    A class to store and manage gradient operator rules.
    """
    # Default static gradient operator order
    _gradient_order = [
        ["PTT", "FOA", "RGT"],
        ["IAD", "RAD", "FD", "IGA"],
        ["CG", "NS"],
    ]

    @staticmethod
    def get_gradient_order() -> List[List[str]]:
        """
        Get the current gradient operator order.

        Returns
        -------
        List[List[str]]
            The current gradient operator order.
        """
        return HyperGradientRules._gradient_order

    @staticmethod
    def set_gradient_order(new_order: List[List[str]]):
        """
        Set a new gradient operator order.

        Parameters
        ----------
        new_order : List[List[str]]
            The new gradient operator order to set.

        Raises
        ------
        ValueError
            If the new order is invalid.
        """
        if not isinstance(new_order, list) or not all(isinstance(group, list) for group in new_order):
            raise ValueError("Gradient order must be a list of lists.")
        HyperGradientRules._gradient_order = new_order


def l2_reg(parameters):
    """
    Compute the L2 regularization loss for a list of parameters.

    :param parameters: Model parameters for which the L2 regularization is computed.
    :type parameters: Iterable[jittor.Var]

    :returns: L2 regularization loss.
    :rtype: jittor.Var
    """
    loss = 0
    for w in parameters:
        loss += (w**2).sum()  # Jittor-compatible L2 norm
    return loss


def grad_unused_zero(output, inputs, retain_graph=False):
    """
    Compute gradients for inputs with respect to the output, filling missing gradients with zeros.

    :param output: The output tensor to compute gradients for.
    :type output: jittor.Var

    :param inputs: The input tensors to compute gradients with respect to.
    :type inputs: List[jittor.Var]

    :param retain_graph: Whether to retain the computation graph. Default is False.
    :type retain_graph: bool

    :returns: Gradients with respect to the inputs, with zeros for unused gradients.
    :rtype: Tuple[jittor.Var]
    """
    grads = jit.grad(output, inputs, retain_graph=retain_graph)

    def grad_or_zeros(grad, var):
        # Replace None or NaN gradients with zeros
        return jit.zeros_like(var) if grad is None or (jit.isnan(grad).any()) else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def list_tensor_matmul(list1, list2):
    """
    Compute the element-wise multiplication and sum of two lists of tensors.

    :param list1: The first list of tensors.
    :type list1: List[jittor.Var]

    :param list2: The second list of tensors.
    :type list2: List[jittor.Var]

    :returns: The resulting scalar from element-wise multiplication and summation.
    :rtype: jittor.Var
    """
    out = 0
    for t1, t2 in zip(list1, list2):
        out += (t1 * t2).sum()  # Element-wise multiplication and sum
    return out


def list_tensor_norm(list_tensor, p=2):
    """
    Compute the p-norm of a list of tensors.

    :param list_tensor: The list of tensors to compute the norm for.
    :type list_tensor: List[jittor.Var]

    :param p: The order of the norm. Default is 2 (Euclidean norm).
    :type p: float

    :returns: The computed p-norm of the list of tensors.
    :rtype: jittor.Var

    :raises ValueError: If the list of tensors is empty.
    """
    norm = 0
    for t in list_tensor:
        # Compute the p-norm for each tensor and accumulate
        norm += (
            t.abs() ** p
        ).sum()  # Element-wise absolute value raised to the power of p
    return norm ** (1 / p)  # Take the p-th root of the accumulated sum


def require_model_grad(model=None):
    """
    Ensure all model parameters require gradients.

    :param model: The model to check and update parameters.
    :type model: jittor.Module

    :raises AssertionError: If the model is not defined.
    """
    assert model is not None, "The module is not defined!"
    for param in model.parameters():
        if param.is_stop_grad():  # Jittor-specific check for non-gradient variables
            param = param.clone()  # Recreate the variable to enable gradients


def update_grads(grads, model):
    """
    Update the custom_grad attribute of the model's parameters.

    :param grads: Gradients to be applied to the parameters.
    :type grads: List[jittor.Var]

    :param model: Model whose parameters will be updated.
    :type model: jittor.Module
    """
    for p, x in zip(grads, model.parameters()):
        if not hasattr(x, "_custom_grad"):
            # Initialize _custom_grad if it doesn't exist
            x._custom_grad = p.clone()
        else:
            # Accumulate gradients in _custom_grad
            x._custom_grad += p


def manual_update(optimizer, variables: List[jit.Var]):
    """
    Manually update variables using gradients stored in _custom_grad.

    :param optimizer: The Jittor optimizer instance.
    :type optimizer: jittor.optim.Optimizer

    :param variables: A list of Jittor variables to be updated.
    :type variables: List[jittor.Var]

    :raises AttributeError: If a variable does not have the '_custom_grad' attribute.
    """
    for group in optimizer.param_groups:
        lr = group.get(
            "lr", optimizer.lr
        )  # Get the learning rate from the optimizer or group

        for param in group["params"]:
            if param in variables:
                # Check if the gradient is available
                if not hasattr(param, "_custom_grad"):
                    raise AttributeError(
                        f"Variable '{param.name}' does not have '_custom_grad'. "
                        f"Ensure gradients are precomputed and stored before updating."
                    )

                grad = param._custom_grad

                # Update the variable using the gradient and learning rate
                param -= lr * grad

                # Optional: Reset _custom_grad if needed
                param._custom_grad *= 0  # Reset for the next iteration


def update_tensor_grads(hparams, grads):
    """
    Update gradients for Jittor variables manually.

    :param hparams: List of Jittor variables representing the hyperparameters.
    :type hparams: List[jittor.Var]

    :param grads: List of gradients corresponding to the hyperparameters.
    :type grads: List[jittor.Var

    :raises ValueError: If a variable is stop_grad and cannot be updated.
    """
    for l, g in zip(hparams, grads):
        # 手动设置 jittor.Var 的 grad
        if l.is_stop_grad():
            raise ValueError(f"Variable {l.name()} is stop_grad and cannot be updated.")
        if not hasattr(l, "_custom_grad"):
            # 初始化自定义梯度
            l._custom_grad = g.clone().detach()
        else:
            # 累加梯度
            l._custom_grad += g


def stop_grads(grads):
    """
    Detach and stop gradient computation for a list of gradients.

    :param grads: The gradients to process.
    :type grads: List[jittor.Var]

    :returns: Detached gradients with requires_grad set to False.
    :rtype: List[jittor.Var
    """
    return [(grad.detach().stop_grad() if grad is not None else grad) for grad in grads]


def average_grad(model, batch_size):
    """
    Divide the gradients of all model parameters by the batch size.

    :param model: The model whose gradients need to be averaged.
    :type model: jittor.Module

    :param batch_size: The batch size to divide gradients by.
    :type batch_size: int
    """
    for param in model.parameters():
        if param.opt_grad() is not None:
            param.opt_grad().update(param.opt_grad() / batch_size)


def stop_model_grad(model=None):
    """
    Stop gradient computation for all parameters in a model.

    :param model: The model to stop gradients for.
    :type model: jittor.Module
    """
    assert model is not None, "The module is not defined!"
    for param in model.parameters():
        param.stop_grad()


def cat_list_to_tensor(list_tx):
    """
    Concatenate a list of tensors into a single flattened tensor.

    :param list_tx: The list of tensors to concatenate.
    :type list_tx: List[jittor.Var]

    :returns: A single flattened tensor.
    :rtype: jittor.Var
    """
    return jit.concat([xx.flatten() for xx in list_tx])


def copy_parameter_from_list(y, z):
    """
    Copy parameters from a list to the parameters of a Jittor model.

    :param y: Jittor model with parameters to be updated.
    :type y: jittor.Module

    :param z: List of variables to copy from.
    :type z: List[jittor.Var

    :returns: Updated model.
    :rtype: jittor.Module
    """
    for p, q in zip(y.parameters(), z):
        # 更新 p 的数据为 q 的克隆
        p.update(q.clone().detach())
        # 确保 p 的 requires_grad 状态保持为 True
        p.requires_grad = True
    return y


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    """
    Compute gradients of the outer-level loss with respect to parameters and hyperparameters.

    :param outer_loss: The scalar loss from the outer-level optimization problem. Typically computed
        from a validation set or during a separate evaluation phase.
    :type outer_loss: jittor.Var

    :param params: The list of parameters for which gradients with respect to the outer loss are computed.
    :type params: List[jittor.Var]

    :param hparams: The list of hyperparameters for which gradients with respect to the outer loss are computed.
    :type hparams: List[jittor.Var]

    :param retain_graph: Whether to retain the computation graph after computing the gradients. Default is True.
    :type retain_graph: bool

    :returns: A tuple containing two lists:
        - `grad_outer_w`: Gradients of the outer loss with respect to `params`.
        - `grad_outer_hparams`: Gradients of the outer loss with respect to `hparams`.
    :rtype: Tuple[List[jittor.Var], List[jittor.Var]]

    :notes:
        - The `grad_unused_zero` utility ensures that any parameter or hyperparameter without a valid gradient
          is assigned a zero tensor as its gradient.
        - Ensure that the `outer_loss` is a scalar tensor (i.e., a single value) for proper gradient computation.
    """

    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(
        outer_loss, hparams, retain_graph=retain_graph
    )

    return grad_outer_w, grad_outer_hparams


def custom_grad(outputs, inputs, grad_outputs=None, retain_graph=False):
    """
    Compute the vector-Jacobian product for Jittor, mimicking PyTorch's autograd.grad.

    :param outputs: Outputs of the differentiated function.
    :type outputs: Sequence[jittor.Var]

    :param inputs: Inputs with respect to which the gradient will be computed.
    :type inputs: Sequence[jittor.Var]

    :param grad_outputs: Gradients with respect to the outputs.
    :type grad_outputs: Sequence[jittor.Var or None

    :param retain_graph: Whether to retain the computation graph after computing the gradients.
    :type retain_graph: bool

    :returns: Gradients with respect to the inputs.
    :rtype: List[jittor.Var]
    """
    # Ensure outputs and grad_outputs are tuples
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    if grad_outputs is None:
        grad_outputs = [jit.ones_like(output) for output in outputs]
    elif not isinstance(grad_outputs, (tuple, list)):
        grad_outputs = (grad_outputs,)

    assert len(outputs) == len(
        grad_outputs
    ), "outputs and grad_outputs must have the same length."

    # Compute the weighted scalar output for gradient computation
    total_output = sum(
        (output * grad_output).sum()
        for output, grad_output in zip(outputs, grad_outputs)
    )

    # Calculate gradients with respect to inputs
    grads = jit.grad(total_output, inputs, retain_graph=retain_graph)

    return grads


def neumann(
    params: List[jit.Var],
    hparams: List[jit.Var],
    upper_loss,
    lower_loss,
    k: int,
    fp_map: Callable[[List[jit.Var], List[jit.Var]], List[jit.Var]],
    tol=1e-10,
) -> List[jit.Var]:
    """
    Compute hyperparameter gradients using the Neumann series approximation.

    :param params: List of parameters for the lower-level optimization problem.
    :type params: List[jit.Var]

    :param hparams: List of hyperparameters for the upper-level optimization problem.
    :type hparams: List[jit.Var]

    :param upper_loss: Loss function for the upper-level problem.
    :type upper_loss: jit.Var

    :param lower_loss: Loss function for the lower-level problem.
    :type lower_loss: jit.Var

    :param k: Number of iterations for the Neumann series approximation.
    :type k: int

    :param fp_map: Fixed-point map function that computes updates to lower-level parameters.
    :type fp_map: Callable[[List[jit.Var], List[jit.Var]], List[jit.Var]]

    :param tol: Tolerance for early stopping based on convergence. Default is 1e-10.
    :type tol: float

    :returns: Hyperparameter gradients computed using the Neumann series approximation.
    :rtype: List[jit.Var]
    """

    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)

    w_mapped = fp_map(params, lower_loss)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for i in range(k):
        gs_prev_vec = gs_vec
        vs = custom_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(jit.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = custom_grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    return grads


def conjugate_gradient(
    params: List[jit.Var],
    hparams: List[jit.Var],
    upper_loss,
    lower_loss,
    K: int,
    fp_map: Callable[[List[jit.Var], List[jit.Var]], List[jit.Var]],
    tol=1e-10,
    stochastic=False,
) -> List[jit.Var]:
    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)

    """
    Compute hyperparameter gradients using the Conjugate Gradient method.

    :param params: List of parameters for the lower-level optimization problem.
    :type params: List[jit.Var]

    :param hparams: List of hyperparameters for the upper-level optimization problem.
    :type hparams: List[jit.Var]

    :param upper_loss: Loss function for the upper-level problem.
    :type upper_loss: jit.Var

    :param lower_loss: Loss function for the lower-level problem.
    :type lower_loss: jit.Var

    :param K: Maximum number of iterations for the Conjugate Gradient method.
    :type K: int

    :param fp_map: Fixed-point map function that computes updates to lower-level parameters.
    :type fp_map: Callable[[List[jit.Var], List[jit.Var]], List[jit.Var]]

    :param tol: Tolerance for early stopping based on convergence. Default is 1e-10.
    :type tol: float

    :param stochastic: If True, recompute the fixed-point map during each iteration. Default is False.
    :type stochastic: bool

    :returns: Hyperparameter gradients computed using the Conjugate Gradient method.
    :rtype: List[jit.Var]
    """

    if not stochastic:
        w_mapped = fp_map(params, lower_loss)

    def dfp_map_dw(xs):
        if stochastic:
            w_mapped_in = fp_map(params, lower_loss)
            Jfp_mapTv = custom_grad(
                w_mapped_in, params, grad_outputs=xs, retain_graph=False
            )
        else:
            Jfp_mapTv = custom_grad(
                w_mapped, params, grad_outputs=xs, retain_graph=True
            )
        return [v - j for v, j in zip(xs, Jfp_mapTv)]

    vs = cg_step(
        dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol
    )  # K steps of conjugate gradient

    if stochastic:
        w_mapped = fp_map(params, lower_loss)

    grads = custom_grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]

    return grads


def cg_step(Ax, b, max_iter=100, epsilon=1.0e-5):
    """
    Perform Conjugate Gradient (CG) optimization to solve Ax = b.

    :param Ax: Function that computes the matrix-vector product Ax for a given x.
    :type Ax: Callable[[List[jit.Var]], List[jit.Var]]

    :param b: Right-hand side of the equation Ax = b.
    :type b: List[jit.Var]

    :param max_iter: Maximum number of iterations for the CG method. Default is 100.
    :type max_iter: int

    :param epsilon: Convergence threshold for the residual norm. Default is 1e-5.
    :type epsilon: float

    :returns: Solution vector x that approximately solves Ax = b.
    :rtype: List[jit.Var]
    """

    x_last = [jit.zeros_like(bb) for bb in b]
    r_last = [jit.zeros_like(bb) + bb for bb in b]  # Jittor does not have copy_, use +
    p_last = [
        jit.zeros_like(rr) + rr for rr in r_last
    ]  # Jittor does not have copy_, use +

    for _ in range(max_iter):
        Ap = Ax(p_last)
        Ap_vec = cat_list_to_tensor(Ap)
        p_last_vec = cat_list_to_tensor(p_last)
        r_last_vec = cat_list_to_tensor(r_last)
        rTr = jit.sum(r_last_vec * r_last_vec)
        pAp = jit.sum(p_last_vec * Ap_vec)
        alpha = rTr / pAp

        x = [xx + alpha * pp for xx, pp in zip(x_last, p_last)]
        r = [rr - alpha * pp for rr, pp in zip(r_last, Ap)]
        r_vec = cat_list_to_tensor(r)

        if float(jit.norm(r_vec)) < epsilon:
            break

        beta = jit.sum(r_vec * r_vec) / rTr
        p = [rr + beta * pp for rr, pp in zip(r, p_last)]

        x_last = x
        p_last = p
        r_last = r

    return x_last
