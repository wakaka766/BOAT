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

def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)
    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None or (torch.isnan(grad).any()) else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))

def list_tensor_matmul(list1,list2,trans=0):
    out=0
    for t1,t2 in zip(list1,list2):
        out=out+torch.sum(t1*t2)
    return out


def list_tensor_norm(list,p=2):
    norm=0
    for t in list:
        norm=norm+torch.norm(t,p)
    return norm


def require_model_grad(model=None):
    assert model is not None, 'The module is not defined!'
    for param in model.parameters():
        param.requires_grad_(True)

def classification_acc(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item() / len(target)

def stop_grad_fn(all_grads):
    for grad in all_grads:
        grad.requires_grad_(False)

def update_grads(grads, model):
    for p, x in zip(grads, model.parameters()):
        if x.grad is None:
            x.grad = p
        else:
            x.grad += p

def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = g
        else:
            l.grad += g

# def stop_grads(grads):
#     for grad in grads:
#         grad = grad.detach()
#         grad.requires_grad = False
#     return grads

def stop_grads(grads):
    return [grad.detach().requires_grad_(False) for grad in grads]
def average_grad(model,batch_size):
    for param in model.parameters():
        param.grad.data= param.grad.data/batch_size


def stop_model_grad(model=None):
    assert model is not None, 'The module is not defined!'
    for param in model.parameters():
        param.requires_grad_(False)



def copy_parameter_from_list(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


class DifferentiableOptimizer:
    def __init__(self, loss_f, dim_mult, data_or_iter=None):
        """
        Args:
            loss_f: callable with signature (params, hparams, [data optional]) -> loss tensor
            data_or_iter: (x, y) or iterator over the data needed for loss_f
        """
        self.data_iterator = None
        if data_or_iter:
            self.data_iterator = data_or_iter if hasattr(data_or_iter, '__next__') else repeat(data_or_iter)

        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def get_opt_params(self, params):
        opt_params = [p for p in params]
        opt_params.extend([torch.zeros_like(p) for p in params for _ in range(self.dim_mult-1) ])
        return opt_params

    def step(self, params, hparams, create_graph,only_grad=False):
        raise NotImplementedError

    def __call__(self, params, hparams, create_graph=True,only_grad=False):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph,only_grad=only_grad)

    def get_loss(self, params, hparams):
        if self.data_iterator:
            data = next(self.data_iterator)
            self.curr_loss = self.loss_f(params, hparams, data)
        else:
            self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss


class HeavyBall(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size, momentum, data_or_iter=None):
        super(HeavyBall, self).__init__(loss_f, dim_mult=2, data_or_iter=data_or_iter)
        self.loss_f = loss_f
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size
        self.momentum_f = momentum if callable(momentum) else lambda x: momentum

    def step(self, params, hparams, create_graph):
        n = len(params) // 2
        p, p_aux = params[:n], params[n:]
        loss = self.get_loss(p, hparams)
        sz, mu = self.step_size_f(hparams), self.momentum_f(hparams)
        p_new, p_new_aux = heavy_ball_step(p, p_aux, loss, sz,  mu, create_graph=create_graph)
        return [*p_new, *p_new_aux]


class Momentum(DifferentiableOptimizer):
    """
    GD with momentum step as implemented in torch.optim.SGD
    .. math::
              v_{t+1} = \mu * v_{t} + g_{t+1} \\
              p_{t+1} = p_{t} - lr * v_{t+1}
    """
    def __init__(self, loss_f, step_size, momentum, data_or_iter=None):
        super(Momentum, self).__init__(loss_f, dim_mult=2, data_or_iter=data_or_iter)
        self.loss_f = loss_f
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size
        self.momentum_f = momentum if callable(momentum) else lambda x: momentum

    def step(self, params, hparams, create_graph):
        n = len(params) // 2
        p, p_aux = params[:n], params[n:]
        loss = self.get_loss(p, hparams)
        sz, mu = self.step_size_f(hparams), self.momentum_f(hparams)
        p_new, p_new_aux = torch_momentum_step(p, p_aux, loss, sz,  mu, create_graph=create_graph)
        return [*p_new, *p_new_aux]


class GradientDescent(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size, data_or_iter=None):
        super(GradientDescent, self).__init__(loss_f, dim_mult=1, data_or_iter=data_or_iter)
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph,only_grad=False):
        loss = self.get_loss(params, hparams)
        sz = self.step_size_f(hparams)
        if only_grad:
            grad0=torch.autograd.grad(loss, params, create_graph=create_graph,allow_unused=True)
            return [(g if g is not None else 0*grad0[0]) for g in grad0]
        else:
            return gd_step(params, loss, sz, create_graph=create_graph)


def gd_step(params, loss, step_size, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph,allow_unused=True)

    return [w - step_size * (g if g is not None else 0) for w, g in zip(params, grads)]


def heavy_ball_step(params, aux_params, loss, step_size, momentum, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g + momentum * (w - v) for g, w, v in zip(grads, params, aux_params)], params


def torch_momentum_step(params, aux_params, loss, step_size, momentum, create_graph=True):
    """
    GD with momentum step as implemented in torch.optim.SGD
    .. math::
              v_{t+1} = \mu * v_{t} + g_{t+1} \\
              p_{t+1} = p_{t} - lr * v_{t+1}
    """
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    new_aux_params = [momentum*v + g for v, g in zip(aux_params, grads)]
    return [w - step_size * nv for w, nv in zip(params, new_aux_params)], new_aux_params
