import argparse
import numpy as np

import os
# 根据您系统的环境变量名称，确保这两个变量被设置
# os.environ['CUDA_PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1'
# os.environ['CUDA_PATH_V12_1'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1'
#
# # 在Jittor的路径设置中，确保这两个变量在PATH中
# os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin;C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\libnvvp'
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# import os
# os.environ["CUDNN_INCLUDE_DIR"] = r"C:\Users\ASUS\.cache\jittor\jtcuda\cuda11.2_cudnn8_win\include"
# os.environ["CUDNN_LIB_DIR"] = r"C:\Users\ASUS\.cache\jittor\jtcuda\cuda11.2_cudnn8_win\lib\x64"
# os.environ["PATH"] = r"C:\Users\ASUS\.cache\jittor\jtcuda\cuda11.2_cudnn8_win\lib\x64" + ";" + os.environ["PATH"]

import jittor as jit
print("Jittor CUDA version:", jit.compiler.cuda_version)

import boat_jit as boat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups_vectorized
from torchvision import datasets
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




# def get_data(args):
#     def from_sparse(x):
#         x = x.tocoo()
#         values = x.data
#         indices = np.vstack((x.row, x.col))
#         i = torch.LongTensor(indices)
#         v = torch.FloatTensor(values)
#         shape = x.shape
#         return torch.sparse.FloatTensor(i, v, torch.Size(shape))
#
#     val_size = 0.5
#     train_x, train_y = fetch_20newsgroups_vectorized(subset='train',
#                                                      return_X_y=True,
#                                                      data_home=args.data_path,
#                                                      download_if_missing=True)
#
#     test_x, test_y = fetch_20newsgroups_vectorized(subset='test',
#                                                    return_X_y=True,
#                                                    data_home=args.data_path,
#                                                    download_if_missing=True)
#
#     train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, stratify=train_y, test_size=val_size)
#     test_x, teval_x, test_y, teval_y = train_test_split(test_x, test_y, stratify=test_y, test_size=0.5)
#
#     train_x, val_x, test_x, teval_x = map(from_sparse, [train_x, val_x, test_x, teval_x])
#     train_y, val_y, test_y, teval_y = map(torch.LongTensor, [train_y, val_y, test_y, teval_y])
#
#     print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])
#     return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)

def get_data(args):
    """
    Load and process data for Jittor. It converts sparse matrices to dense tensors
    and splits the dataset into training, validation, test, and evaluation sets.

    Args:
        args: Argument object containing the data path.

    Returns:
        tuple: Contains training, validation, test, and evaluation datasets
               in the form (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y).
    """

    def from_sparse(x):
        """
        Convert a scipy sparse matrix to a Jittor dense tensor.

        Args:
            x (scipy.sparse matrix): Input sparse matrix.

        Returns:
            jittor.Var: Dense tensor corresponding to the input sparse matrix.
        """
        x = x.tocoo()  # Convert to COOrdinate format
        values = x.data  # Non-zero values of the sparse matrix
        indices = np.vstack((x.row, x.col))  # Stack the row and column indices

        # Replace torch.LongTensor with jit.array of int64
        i = jit.array(indices, dtype=jit.int64)  # Indices of non-zero elements
        v = jit.array(values, dtype=jit.float32)  # Values of non-zero elements

        shape = x.shape  # Shape of the sparse matrix

        # Create a dense tensor from the indices and values
        dense_tensor = jit.zeros(shape, dtype=jit.float32)
        dense_tensor[i[0], i[1]] = v  # Place non-zero elements at the correct positions
        return dense_tensor

    val_size = 0.5  # Proportion of data to be used for validation

    # Load the training and testing datasets
    train_x, train_y = fetch_20newsgroups_vectorized(subset='train',
                                                     return_X_y=True,
                                                     data_home=args.data_path,
                                                     download_if_missing=True)

    test_x, test_y = fetch_20newsgroups_vectorized(subset='test',
                                                   return_X_y=True,
                                                   data_home=args.data_path,
                                                   download_if_missing=True)

    # Split the training data into training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, stratify=train_y, test_size=val_size)

    # Split the test data into test and evaluation sets
    test_x, teval_x, test_y, teval_y = train_test_split(test_x, test_y, stratify=test_y, test_size=0.5)

    # Convert the sparse matrices to dense tensors using the from_sparse function
    train_x, val_x, test_x, teval_x = map(from_sparse, [train_x, val_x, test_x, teval_x])

    # Convert labels to Jittor tensors (jit.int64 for integer labels)
    train_y, val_y, test_y, teval_y = map(lambda y: jit.array(y, dtype=jit.int64), [train_y, val_y, test_y, teval_y])

    # Print the sizes of the splits
    print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])

    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)


import os
import json

# base_folder = os.path.dirname(os.path.abspath(__file__))
# 获取上一级路径

# def evaluate(x, w, testset):
#     with torch.no_grad():
#         test_x, test_y = testset
#         y = test_x.mm(x)
#         loss = nn.cross_entropy_loss(y, test_y).detach().item()
#         acc = (y.argmax(-1).eq(test_y).sum() / test_y.shape[0]).detach().cpu().item()
#     return loss, acc

def evaluate(x, w, testset):
    """
    Evaluate the performance of the model on the test set.

    Args:
        x (jittor.Var): Input data tensor.
        w (jittor.Var): Model weights.
        testset (tuple): Tuple containing test_x and test_y.

    Returns:
        tuple: Loss and accuracy of the model on the test set.
    """
    with jit.no_grad():  # Disable gradient calculation
        test_x, test_y = testset  # Unpack the test set
        y = test_x @ x  # Matrix multiplication
        loss = jit.nn.cross_entropy_loss(y, test_y).detach().item()  # Calculate cross-entropy loss
        acc = (y.argmax(-1) == test_y).sum().item() / test_y.shape[0]  # Calculate accuracy
    return loss, acc



base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)

with open(os.path.join(parent_folder, "configs_jit/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(parent_folder, "configs_jit/loss_config_l2.json"), "r") as f:
    loss_config = json.load(f)


def main():
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--generate_data', action='store_true',
                            default=False, help='whether to create data')
        parser.add_argument('--pretrain', action='store_true',
                            default=False, help='whether to create data')
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--iterations', type=int, default=10, help='T')
        parser.add_argument('--data_path', default='./data', help='where to save data')
        parser.add_argument('--model_path', default='./save_l2reg', help='where to save model')
        parser.add_argument('--x_lr', type=float, default=100)
        parser.add_argument('--xhat_lr', type=float, default=100)
        parser.add_argument('--w_lr', type=float, default=1000)

        parser.add_argument('--w_momentum', type=float, default=0.9)
        parser.add_argument('--x_momentum', type=float, default=0.9)

        parser.add_argument('--K', type=int, default=10, help='k')

        parser.add_argument('--u1', type=float, default=1.0)
        parser.add_argument('--BVFSM_decay', type=str, default='log', choices=['log', 'power2'])
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--alg', type=str, default='BOME', choices=[
            'BOME', 'BSG_1', 'penalty', 'AID_CG', 'AID_FP', 'ITD', 'BVFSM', 'baseline', 'VRBO', 'reverse', 'stocBiO',
            'MRBO']
                            )
        parser.add_argument('--dynamic_method', type=str, default=None,
                            help='omniglot or miniimagenet or tieredImagenet')
        parser.add_argument('--hyper_method', type=str, default=None,
                            help='convnet for 4 convs or resnet for Residual blocks')
        parser.add_argument('--fo_gm', type=str, default=None, help='convnet for 4 convs or resnet for Residual blocks')
        args = parser.parse_args()

        np.random.seed(args.seed)
        jit.set_global_seed(args.seed)
        return args

    args = parse_args()
    trainset, valset, testset, tevalset = get_data(args)
    # torch.save((trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg.pt"))
    # print(f"[info] successfully generated data to {args.data_path}/l2reg.pt")
    # device = torch.device("cpu") ## torch.device("cuda") if torch.cuda.is_available() else
    # n_feats = trainset[0].shape[-1]
    # num_classes = trainset[1].unique().shape[-1]
    #
    # class upper_model(torch.nn.Module):
    #
    #     def __init__(self, n_feats, device):
    #         super(upper_model, self).__init__()
    #         self.x = torch.nn.Parameter(torch.zeros(n_feats, requires_grad=True, device=device).requires_grad_(True))
    #
    #     def forward(self):
    #         return self.x
    #
    # class lower_model(torch.nn.Module):
    #
    #     def __init__(self, n_feats, device):
    #         super(lower_model, self).__init__()
    #         self.y = torch.nn.Parameter(torch.zeros((n_feats, num_classes), requires_grad=True, device=device))
    #         self.y.data = nn.init.kaiming_normal_(self.y.data.t(), mode='fan_out').t()
    #         # self.y.data.copy_(torch.load("./save_l2reg/pretrained.pt").to(args.device))
    #
    #     def forward(self):
    #         return self.y

    jit.save((trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg.pkl"))
    print(f"[info] successfully generated data to {args.data_path}/l2reg.pkl")

    class UpperModel(jit.Module):
        def __init__(self, n_feats):
            self.x = jit.init.constant([n_feats], 0.0).stop_grad()

    class LowerModel(jit.Module):
        def __init__(self, n_feats, num_classes):
            self.y = jit.init.kaiming_normal([n_feats, num_classes])

    upper_model = UpperModel(trainset[0].shape[-1])
    lower_model = LowerModel(trainset[0].shape[-1], int(trainset[1].max().item()) + 1)
    upper_opt = jit.nn.Adam(upper_model.parameters(), lr=0.01)
    lower_opt = jit.nn.SGD(lower_model.parameters(), lr=0.01)

    for x_itr in range(1000):
        loss, run_time = 0, 0  # Placeholder logic for loss and runtime computation
        print(f"[info] epoch {x_itr:5d} loss {loss:10.4f} time {run_time:8.2f}")

    # upper_model = upper_model(n_feats, device)
    # lower_model = lower_model(n_feats, device)
    # upper_opt = torch.optim.Adam(upper_model.parameters(), lr=0.01)
    # lower_opt = torch.optim.SGD(lower_model.parameters(), lr=0.01)
    print(args.dynamic_method)
    print(args.hyper_method)
    dynamic_method = args.dynamic_method.split(',') if args.dynamic_method else []
    hyper_method = args.hyper_method.split(',') if args.hyper_method else []
    if "RGT" in hyper_method:
        boat_config['RGT']['truncate_iter'] = 1
    boat_config["dynamic_op"] = dynamic_method
    boat_config["hyper_op"] = hyper_method
    boat_config["fo_gm"] = args.fo_gm
    boat_config['lower_level_model'] = lower_model
    boat_config['upper_level_model'] = upper_model
    boat_config['lower_level_var'] = lower_model.parameters()
    boat_config['upper_level_var'] = upper_model.parameters()
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver(lower_opt)
    b_optimizer.build_ul_solver(upper_opt)

    ul_feed_dict = {"data": trainset[0], "target": trainset[1]}
    ll_feed_dict = {"data": valset[0], "target": valset[1]}

    # ul_feed_dict = {"data": trainset[0].to(device), "target": trainset[1].to(device)}
    # ll_feed_dict = {"data": valset[0].to(device), "target": valset[1].to(device)}

    if "DM" in boat_config["dynamic_op"] and ("GDA" in boat_config["dynamic_op"]):
        iterations = 3000
    else:
        iterations = 1000
    for x_itr in range(iterations):
        if "DM" in boat_config["dynamic_op"] and ("GDA" in boat_config["dynamic_op"]):
            b_optimizer._ll_solver.strategy = "s" + str(x_itr + 1)
        loss, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)

        if x_itr % 1 == 0:
            test_loss, test_acc = evaluate(lower_model(), upper_model(), testset)
            teval_loss, teval_acc = evaluate(lower_model(), upper_model(), tevalset)
            print(
                f"[info] epoch {x_itr:5d} te loss {test_loss:10.4f} te acc {test_acc:10.4f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {run_time:8.2f}")


if __name__ == '__main__':
    main()
