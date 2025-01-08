import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

from model_search import Network, Architecture
import utils

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--load', action='store_true', default=False, help='Whether to load model')
parser.add_argument('--load_iter', type=int, default=0, help='the itrations to load the model')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--inner_loop', type=int, default=1, help='number of inner loops for IAPTT')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = os.path.join('NAS', args.save)
if not os.path.exists('NAS'):
    os.makedirs('NAS')

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

np.random.seed(args.seed)
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled = True
torch.cuda.manual_seed(args.seed)
logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
model = model.cuda()
if args.load:
    utils.load(model, args.model_path)
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

train_transform, valid_transform = utils._data_transforms_cifar10(args)
train_data = dset.CIFAR10(
    root=args.data, train=True, download=True, transform=train_transform
)
valid_data = dset.CIFAR10(
    root=args.data, train=False, download=True, transform=valid_transform
)

test_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=2
)
num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))
train_iters = int(
    args.epochs
    * (num_train * args.train_portion // args.batchsz + 1)
    * args.unroll_steps
)

arch_net = Architecture(steps=args.arch_steps)
arch_optimizer = optim.Adam(
    arch_net.parameters(),
    lr=args.arch_lr,
    betas=(0.5, 0.999),
    weight_decay=args.arch_wd,
)
arch_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batchsz,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    pin_memory=True,
    num_workers=2,
)

criterion = nn.CrossEntropyLoss()
classifier_net = Network(
    args.init_ch, 10, args.layers, criterion, steps=args.arch_steps
)
classifier_optimizer = torch.optim.SGD(
    classifier_net.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd,
)

classifier_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batchsz,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True,
    num_workers=0,
)

class Arch(ImplicitProblem):
    def training_step(self, batch):
        x, target = batch
        alphas = self.forward()
        loss = self.classifier.module.loss(x, alphas, target)

        return loss

class Classifier(ImplicitProblem):
    def training_step(self, batch):
        x, target = batch
        alphas = self.arch()
        loss = self.module.loss(x, alphas, target)

        return loss

class NASEngine(Engine):
    @torch.no_grad()
    def validation(self):
        corrects = 0
        total = 0
        for x, target in test_queue:
            x, target = x.to(device), target.to(device, non_blocking=True)
            alphas = self.arch()
            _, correct = self.classifier.module.loss(x, alphas, target, acc=True)
            corrects += correct
            total += x.size(0)
        acc = corrects / total

        alphas = self.arch()
        torch.save({"genotype": self.classifier.module.genotype(alphas)}, "genotype.t7")
        return {"acc": acc}


outer_config = Config(retain_graph=True)
inner_config = Config(type="darts", unroll_steps=args.unroll_steps)
engine_config = EngineConfig(
    valid_step=args.report_freq * args.unroll_steps,
    train_iters=train_iters,
    roll_back=True,
)
outer = Arch(
    name="arch",
    module=arch_net,
    optimizer=arch_optimizer,
    train_data_loader=arch_loader,
    config=outer_config,
)
inner = Classifier(
    name="classifier",
    module=classifier_net,
    optimizer=classifier_optimizer,
    scheduler=classifier_scheduler,
    train_data_loader=classifier_loader,
    config=inner_config,
)

problems = [outer, inner]
l2u = {inner: [outer]}
u2l = {outer: [inner]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()

