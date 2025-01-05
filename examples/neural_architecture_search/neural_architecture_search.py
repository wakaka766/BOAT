import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import boat
from torch.autograd import Variable
from model_search import Network



parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
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
parser.add_argument('--load_iter',type=int,default=0,help='the itrations to load the model')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--inner_loop', type=int, default=1, help='number of inner loops for IAPTT')
parser.add_argument('--cg_steps', type=int, default=1, help='number of cg loops for IGBR')
parser.add_argument('--rhg', action='store_true', default=False, help='Whether to use RHG')
parser.add_argument('--bamm', action='store_true', default=False, help='Whether to use bammgon')
parser.add_argument('--cg', action='store_true', default=False, help='Whether to use CG')
parser.add_argument('--lamb', type=float, default=0.1, help='lambda for CG')
parser.add_argument('--alpha', type=float, default=0.0, help='\mu, coefficient')
parser.add_argument('--gamma2', type=float, default=0.00001)#0.00001 for BRMM
parser.add_argument('--gamma1', type=float, default=0.5)#0.5 for BRMM
parser.add_argument('--p', type=float, default=14)
parser.add_argument('--ita_0', type=float, default=0.05)
parser.add_argument('--ita_bamm', type=float, default=0.05)
parser.add_argument('--mu0', type=float, default=0.89)
parser.add_argument('--tau', type=float, default=1/40.)
parser.add_argument('--BDA', action='store_true', default=False, help='whether aggregate the LL and UL loss')
parser.add_argument('--iaptt', action='store_true', default=False, help='Whether to use IAPTT')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
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

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  optimizer_z = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  outer_opt = torch.optim.Adam(model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  import json
  with open("configs/boat_config.json", "r") as f:
    boat_config = json.load(f)

  with open("configs/loss_config.json", "r") as f:
    loss_config = json.load(f)

  boat_config['lower_level_model'] = model
  boat_config['upper_level_model'] = model
  boat_config['lower_level_var'] = model.parameters()
  boat_config['upper_level_var'] = model.arch_parameters()
  b_optimizer = boat.Problem(boat_config, loss_config)
  b_optimizer.build_ll_solver(optimizer)
  b_optimizer.build_ul_solver(outer_opt)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  if args.iaptt:
    scheduler_z = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer_z, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  # architect = Architect(model, args)

  for epoch in range(args.load_iter, args.epochs):
    if args.load_iter>0:
      for j in range(args.load_iter-1):
        scheduler.step()
    if args.iaptt:
      scheduler_z.step()
      lr = scheduler_z.get_lr()[0]
      logging.info('epoch %d z_lr %e', epoch, lr)
    elif args.bamm:
      lr = scheduler.get_lr()[0]
      logging.info('epoch %d lr %e', epoch, lr)
    else:
      scheduler.step()
      lr = scheduler.get_lr()[0]
      logging.info('epoch %d lr %e', epoch, lr)


    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # if epoch ==0:
    #   valid_acc, valid_obj = infer(valid_queue, model, criterion)
    #   logging.info('valid_acc initial%f', valid_acc)
    # training
    # args.alpha = args.alpha * (epoch + 1) / (epoch + 2)
    # args.alpha = args.mu0 * 1 / (epoch + 1) ** (1 / args.p)
    #
    # args.ita_bamm = args.ita_0*args.alpha**(3/2)
    # x_lr = args.arch_learning_rate *args.alpha**(17/2)
    # for params in architect.optimizer.param_groups:
    #   params['lr'] = x_lr
    logging.info(' ita_bamm %f, mu_k %f',args.ita_bamm, args.alpha)
    # if not args.bamm:
    train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer,optimizer_z, lr,epoch,b_optimizer)


    if not args.bamm:
      logging.info('train_acc %f , train_obj %f', train_acc,train_obj)
    else:
      logging.info('train_acc %f , train_obj %f ', train_acc,train_obj)
    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f, valid_obj %f', valid_acc,valid_obj)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, criterion,optimizer, optimizer_z,lr,epoch,boat_optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()#(async=True)
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()  # cuda(async=True)
    ul_feed_dict = {"data": input_search, "target": target_search}
    ll_feed_dict = {"data": input, "target": target}
    loss, run_time = boat_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=step)

    # architect.step(input, target, input_search, target_search, lr,optimizer,optimizer_z=optimizer_z,
    #                unrolled=args.unrolled,iaptt=args.iaptt,rhg=args.rhg,cg=args.cg,bda=args.BDA,ita_bamm=args.ita_bamm,bamm=args.bamm,inner_loop=args.inner_loop,cg_steps = args.cg_steps,lamb=args.lamb,alpha=args.alpha,gamma1=args.gamma1,gamma2=args.gamma2)

    # optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    # loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    if step >200:
      break
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()#cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    if step >200:
      break

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

