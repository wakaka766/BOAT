from torch.nn import functional as F


def val_loss(val_data, upper_model, lower_model):
    loss = F.cross_entropy(lower_model(val_data['data']), val_data['target'])
    return loss


def train_loss(train_data, upper_model, lower_model):
    out = upper_model(F.cross_entropy(lower_model(train_data['data']), train_data['target'], reduction='none'))
    return out
