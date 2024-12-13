import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import torch.nn.functional as F
import boat
from util_file import data_splitting,initialize
from torchvision.datasets import MNIST
base_folder = os.path.dirname(os.path.abspath(__file__))
dataset = MNIST(root=os.path.join(base_folder,"data/"), train=True, download=True)
tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(0.5)
tr.data_flatten()
val.data_flatten()
test.data_flatten()

def accuary(out, target):
    pred = out.argmax(dim=1, keepdim=True)
    # print(pred)# get the index of the max log-probability
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc


def Binarization(x):
    x_bi = np.zeros_like(x)
    for i in range(x.shape[0]):
        # print(x[i])
        x_bi[i] = 1 if x[i] >= 0 else 0
    return x_bi


print(torch.cuda.is_available())
device=torch.device("cuda")
class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(tr.data.shape[0]).to(device).requires_grad_(True))

    def forward(self, y):
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y
x = Net_x(tr)
y = torch.nn.Sequential(torch.nn.Linear(28 ** 2, 10)).to(device)
x_opt = torch.optim.Adam(x.parameters(), lr=0.01)
y_opt=torch.optim.SGD(y.parameters(), lr=0.01)
###
initialize(x)
initialize(y)
###

import os
import json
base_folder = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_folder,"configs/boat_config_dhl.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(base_folder,"configs/loss_config_dhl.json"), "r") as f:
    loss_config = json.load(f)
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Data HyperCleaner')

    parser.add_argument('--dynamic_method', type=str, default=None,help='omniglot or miniimagenet or tieredImagenet')
    parser.add_argument('--hyper_method', type=str, default=None,help='convnet for 4 convs or resnet for Residual blocks')
    parser.add_argument('--fo_gm', type=str, default=None,help='convnet for 4 convs or resnet for Residual blocks')
    args = parser.parse_args()
    dynamic_method = args.dynamic_method.split(',') if args.dynamic_method else []
    hyper_method = args.hyper_method.split(',') if args.hyper_method else []
    print(args.dynamic_method)
    print(args.hyper_method)
    if "RGT" in hyper_method :
        boat_config['RGT']['truncate_iter']=1
    boat_config["dynamic_op"] =dynamic_method
    boat_config["hyper_op"] = hyper_method
    boat_config["fo_gm"] = args.fo_gm
    boat_config['lower_level_model'] = y
    boat_config['upper_level_model'] = x
    boat_config['lower_level_var'] = y.parameters()
    boat_config['upper_level_var'] = x.parameters()
    b_optimizer = boat.Problem(boat_config,loss_config)
    # b_optimizer = boat.Problem(
    #             'Feature', 'Dynamic', 'RAD',
    #             ll_objective=train_loss, ul_objective=val_loss,
    #             ll_model=y, ul_model=x, total_iters=3000)
    # b_optimizer.build_ll_solver(100, y_opt)
    # b_optimizer.build_ul_solver(x_opt)

    b_optimizer.build_ll_solver(y_opt)
    b_optimizer.build_ul_solver(x_opt)
    ul_feed_dict = {"data": val.data.to(device), "target":val.clean_target.to(device)}
    ll_feed_dict = {"data": tr.data.to(device), "target":tr.dirty_target.to(device)}
    for x_itr in range(2):
        loss, run_time = b_optimizer.run_iter(ll_feed_dict,ul_feed_dict, current_iter=x_itr)

        if x_itr % 1 == 0:
            with torch.no_grad():
                out = y(test.data.to(device))
                acc = accuary(out, test.clean_target.to(device))
                x_bi = Binarization(x.x.cpu().numpy())
                clean = x_bi * tr.clean
                p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                r = clean.mean() / (1. - tr.rho)
                F1_score = 2 * p * r / (p + r + 1e-8)
                dc = 0
                if x_itr==0:
                    F1_score_last=0
                if F1_score_last > F1_score:
                    dc = 1
                F1_score_last = F1_score
                valLoss = F.cross_entropy(out, test.clean_target.to(device))
                print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val_loss={:.3f}'.format(x_itr,
                                                                                 100 * accuary(out,
                                                                                               test.clean_target.to(device)),
                                                                                 100 * p, 100 * r, 100 * F1_score,valLoss))


if __name__ == '__main__':
    main()
