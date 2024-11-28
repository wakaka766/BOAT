import torch
from torch.nn import functional as F


def final_accuary(out, target):
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc


def l2_reg(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def p_norm_reg(parameters, exp, epi):
    loss = 0
    for w in parameters:
        loss += (torch.norm(w, 2)+torch.norm(epi*torch.ones_like(w), 2))**(exp/2)
    return loss


def bias_reg_f(bias, params):
    # l2 biased regularization
    return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])


def distance_reg(output, label, params, hparams, reg_param):
    # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
    return F.cross_entropy(output, label) + reg_param * bias_reg_f(hparams, params)


def classification_acc(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item() / len(target)


def update_grads(grads, model):
    for p, x in zip(grads, model.parameters()):
        if x.grad is None:
            x.grad = p
        else:
            x.grad += p


def copy_parameter_from_list(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y