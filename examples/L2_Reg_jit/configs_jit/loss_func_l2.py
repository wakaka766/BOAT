import jittor as jit
import jittor.nn as nn


def ul_loss(ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    y = jit.matmul(ul_feed_dict["data"], lower_model(**kwargs))
    loss = nn.cross_entropy_loss(y, ul_feed_dict["target"], reduction="mean")
    return loss


def ll_loss(ll_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    y = jit.matmul(ll_feed_dict["data"], lower_model(**kwargs))
    loss = nn.cross_entropy_loss(y, ll_feed_dict["target"], reduction="mean")
    reg_loss = (
        0.5 * (lower_model(**kwargs).pow(2) * upper_model().view(-1, 1).exp()).mean()
    )
    return loss + reg_loss


def gda_loss(
    ll_feed_dict, ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs
):
    y_val = jit.matmul(ul_feed_dict["data"], lower_model(**kwargs))
    loss_val = nn.cross_entropy_loss(y_val, ul_feed_dict["target"], reduction="mean")
    y_tr = jit.matmul(ll_feed_dict["data"], lower_model(**kwargs))
    loss = nn.cross_entropy_loss(y_tr, ll_feed_dict["target"], reduction="mean")
    reg_loss = (
        0.5 * (lower_model(**kwargs).pow(2) * upper_model().view(-1, 1).exp()).mean()
    )
    out = (
        ll_feed_dict["alpha"] * (loss + reg_loss)
        + (1 - ll_feed_dict["alpha"]) * loss_val
    )
    return out
