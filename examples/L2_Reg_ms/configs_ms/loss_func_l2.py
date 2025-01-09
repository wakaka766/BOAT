import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn


def ul_loss(ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    """
    Compute the upper-level loss.

    Args:
        ul_feed_dict (dict): Dictionary containing 'data' and 'target' for upper-level loss computation.
        upper_model (ms.nn.Cell): Upper-level model (not used in this function but kept for consistency).
        lower_model (ms.nn.Cell): Lower-level model.
        weights (float): Weight parameter, default is 0.0.
        **kwargs: Additional arguments for lower_model.

    Returns:
        ms.Tensor: The computed upper-level loss.
    """
    data = ul_feed_dict["data"]
    target = ul_feed_dict["target"]

    if isinstance(data, ms.COOTensor):
        # Use sparse-dense multiplication for COOTensor
        dense_weights = lower_model(**kwargs)
        y = ops.SparseTensorDenseMatmul()(
            data.indices, data.values, data.shape, dense_weights
        )
    else:
        # Use dense MatMul for standard tensors
        y = ops.MatMul()(data, lower_model(**kwargs))

    # Compute softmax cross-entropy loss
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    loss = loss_fn(y, target)

    return loss


def ll_loss(ll_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    """
    Compute the lower-level loss, handling sparse and dense tensor operations.

    Args:
        ll_feed_dict (dict): Dictionary containing 'data' and 'target' for lower-level loss computation.
        upper_model (ms.nn.Cell): Upper-level model.
        lower_model (ms.nn.Cell): Lower-level model.
        weights (float): Weight parameter, default is 0.0.
        **kwargs: Additional arguments for lower_model.

    Returns:
        ms.Tensor: The computed lower-level loss including L2 regularization.
    """
    data = ll_feed_dict["data"]
    target = ll_feed_dict["target"]

    if isinstance(data, ms.COOTensor):
        # Use sparse-dense multiplication for COOTensor
        dense_weights = lower_model(**kwargs)
        y = ops.SparseTensorDenseMatmul()(
            data.indices, data.values, data.shape, dense_weights
        )
    else:
        # Use dense MatMul for standard tensors
        y = ops.MatMul()(data, lower_model(**kwargs))

    # Compute softmax cross-entropy loss
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    loss = loss_fn(y, target)

    # L2 regularization
    reg_loss = (
        0.5
        * (lower_model(**kwargs).pow(2) * ops.Exp()(upper_model().view(-1, 1))).mean()
    )

    return loss + reg_loss


def gda_loss(
    ll_feed_dict, ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs
):
    """
    Compute the Generalized Data Augmentation (GDA) loss.

    Args:
        ll_feed_dict (dict): Dictionary containing 'data', 'target', and 'alpha' for lower-level loss computation.
        ul_feed_dict (dict): Dictionary containing 'data' and 'target' for upper-level loss computation.
        upper_model (ms.nn.Cell): Upper-level model.
        lower_model (ms.nn.Cell): Lower-level model.
        weights (float): Weight parameter, default is 0.0.
        **kwargs: Additional arguments for lower_model.

    Returns:
        ms.Tensor: The computed GDA loss.
    """
    # Handle validation data
    data_val = ul_feed_dict["data"]
    target_val = ul_feed_dict["target"]

    if isinstance(data_val, ms.COOTensor):
        dense_weights = lower_model(**kwargs)
        y_val = ops.SparseTensorDenseMatmul()(
            data_val.indices, data_val.values, data_val.shape, dense_weights
        )
    else:
        y_val = ops.MatMul()(data_val, lower_model(**kwargs))

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    loss_val = loss_fn(y_val, target_val)

    # Handle training data
    data_tr = ll_feed_dict["data"]
    target_tr = ll_feed_dict["target"]

    if isinstance(data_tr, ms.COOTensor):
        dense_weights = lower_model(**kwargs)
        y_tr = ops.SparseTensorDenseMatmul()(
            data_tr.indices, data_tr.values, data_tr.shape, dense_weights
        )
    else:
        y_tr = ops.MatMul()(data_tr, lower_model(**kwargs))

    loss_tr = loss_fn(y_tr, target_tr)

    # L2 regularization
    reg_loss = (
        0.5
        * (lower_model(**kwargs).pow(2) * ops.Exp()(upper_model().view(-1, 1))).mean()
    )

    # Combine losses with alpha weighting
    alpha = ll_feed_dict["alpha"]
    out = alpha * (loss_tr + reg_loss) + (1 - alpha) * loss_val

    return out
