from common import config
from network import Network
import utils

import os
import copy
import random
import argparse
import numpy as np
import hashlib
import subprocess
import datetime
import glob
import logging
from utils import AverageMeter, ProgressMeter

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.cpp_extension

parser = argparse.ArgumentParser(description='Used for resuming training')
parser.add_argument('--resume', default=None, type=int, metavar='ITER',
                    help='done iteration to resume from')
args = parser.parse_args()

# append md5 of cpp file can automatically recompile JIT every
# time we modify cpp file

heads = glob.glob('cpp/*.hpp')
sources = glob.glob('cpp/*.cpp')
hash_code = hashlib.md5(b''.join([open(head, 'rb').read() for head in heads] +
                                 [open(source, 'rb').read() for source in sources])).hexdigest()
sokoban_cpp = torch.utils.cpp_extension.load(name='sokoban_cpp_{}'.format(hash_code),
                                             sources=sources,
                                             extra_cflags=['-std=c++17', '-O6'])

def main():
    config.train_log_dir = utils.get_train_log_dir()

    config.ngpus_per_node = torch.cuda.device_count()
    config.node_rank = int(os.environ['SLURM_PROCID'])
    config.nnodes = int(os.environ['SLURM_NNODES'])

    config.ckpt_dir = os.path.join(config.train_log_dir, 'ckpts')
    config.smod_dir = os.path.join(config.train_log_dir, 'smods')
    if config.node_rank == 0:
        utils.ensure_dir(config.ckpt_dir)
        utils.ensure_dir(config.smod_dir)

    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames'])
    os.environ['MASTER_ADDR'] = hostnames.split()[0].decode('utf-8')
    os.environ['MASTER_PORT'] = str(config.master_port)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    mp.spawn(main_worker, nprocs=config.ngpus_per_node, args=(config,))

def main_worker(gpu, config):
    # In honor of Makoto Soejima
    torch.manual_seed(58)
    random.seed(58)

    cudnn.benchmark = True

    # use nccl backend for gpu data transfer during training
    # use gloo backend for cpu data transfer during dataset sending
    dist.init_process_group(
        backend='nccl',
        rank=config.ngpus_per_node * config.node_rank + gpu,
        world_size=config.ngpus_per_node * config.nnodes,
        timeout=datetime.timedelta(hours=24))

    acc_process_group = dist.new_group(
        backend='gloo',
        timeout=datetime.timedelta(hours=24))

    head_process_group = dist.new_group(
        backend='gloo',
        ranks=[i * config.ngpus_per_node for i in range(config.nnodes)],
        timeout=datetime.timedelta(hours=24))

    node_groups = []
    for i in range(config.nnodes):
        node_groups.append(dist.new_group(
            backend='gloo',
            ranks=[config.ngpus_per_node * i + j for j in range(config.ngpus_per_node)],
            timeout=datetime.timedelta(hours=24)))
    cnt_node_group = node_groups[config.node_rank]

    if dist.get_rank() == 0:
        fh = logging.FileHandler(os.path.join(config.train_log_dir, 'log.txt'), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s[%(levelname)s]: %(message)s'))
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logging.basicConfig(handlers=[fh, ch], level=logging.NOTSET)

    if dist.get_rank() == 0:
        logging.info('Group creation and logging initialization done')

    config.n, config.m, config.npolicy, config.mask, config.nbox = sokoban_cpp.initialize()

    if dist.get_rank() == 0:
        logging.info('Board initialization done')

    device = torch.device('cuda', gpu)
    config.device = device
    torch.cuda.set_device(device)
    model = Network(config.n, config.m,
                    num_embeddings=config.num_embeddings,
                    num_features=config.num_features,
                    num_policies=config.npolicy,
                    num_res_blocks=config.num_res_blocks,
                    mask=config.mask).to(device)

    if args.resume:
        iter_done = args.resume
        if dist.get_rank() == 0:
            logging.info('=> loading from iteration {}'.format(args.resume))
            ckpt = torch.load(os.path.join(config.ckpt_dir, 'iter-{}'.format(args.resume)))
            model.load_state_dict(ckpt['state_dict'])
    else:
        # We initialize a dummy checkpoint at iter 0
        # TODO avoid this code if we can duplicate Script::Module
        # in cpp extension
        iter_done = 0
        if dist.get_rank() == 0:
            save_module(0, model, config)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = torch.optim.AdamW(model.parameters(), 1e+4,
                                  weight_decay=config.weight_decay)

    lo, hi = 2, 2
    acc_tensor = torch.zeros(1)
    best_acc = 0.0
    best_iter = 0

    for it in range(iter_done, config.total_iter):
        if dist.get_rank() == 0:
            logging.info('=> [Iter:{}]\tStarts'.format(it + 1))

        if gpu == 0:
            # all head processes need to wait for rank 0 writing module to the dist
            dist.barrier(group=head_process_group)
            if dist.get_rank() == 0:
                logging.info('=> [Iter:{}]\tSynchronization of head processes done'.format(it + 1))

            data, prob, value, acc, cout = sokoban_cpp.generate_data(
                os.path.join(config.smod_dir, 'iter-{}'.format(it)),
                lo, hi)

            if dist.get_rank() == 0:
                acc_tensor.fill_(acc)

            assert data.size(0) == prob.size(0) and data.size(0) == value.size(0)
            if dist.get_rank() == 0:
                logging.info(cout)
                logging.info('=> [Iter:{}]\tData generation done with size {}'.format(
                             it + 1, data.size(0)))
                logging.info('=> [Iter:{}]\tTotal generated data size {}\tdata sum {:.3f}\tprob sum {:.3f}\tvalue sum {:.3f}'.format(
                             it + 1, data.size(0), data.sum().item(), prob.sum().item(), value.sum().item()))

            ''' now gpu 0 assign jobs to other gpus '''
            ops = []
            for i in range(1, config.ngpus_per_node):
                indexes = torch.randint(data.size(0), size=(config.train_steps * config.batch_size_per_gpu,))
                dst = config.node_rank * config.ngpus_per_node + i
                ops += [dist.isend(data[indexes],
                                   dst=dst,
                                   group=cnt_node_group,
                                   tag=(it*dist.get_world_size()+dst)*3+0),
                        dist.isend(prob[indexes],
                                   dst=dst,
                                   group=cnt_node_group,
                                   tag=(it*dist.get_world_size()+dst)*3+1),
                        dist.isend(value[indexes],
                                   dst=dst,
                                   group=cnt_node_group,
                                   tag=(it*dist.get_world_size()+dst)*3+2)]
            for op in ops:
                op.wait()

            indexes = torch.randint(data.size(0), size=(config.train_steps * config.batch_size_per_gpu,))
            data = data[indexes]
            prob = prob[indexes]
            value = value[indexes]
        else:
            data = torch.empty((config.train_steps * config.batch_size_per_gpu, config.n, config.m), dtype=torch.long)
            prob = torch.empty((config.train_steps * config.batch_size_per_gpu, config.npolicy))
            value = torch.empty((config.train_steps * config.batch_size_per_gpu,))
            ops = [dist.irecv(data, src=config.node_rank * config.ngpus_per_node,
                              group=cnt_node_group,
                              tag=(it*dist.get_world_size()+dist.get_rank())*3+0),
                   dist.irecv(prob, src=config.node_rank * config.ngpus_per_node,
                              group=cnt_node_group,
                              tag=(it*dist.get_world_size()+dist.get_rank())*3+1),
                   dist.irecv(value, src=config.node_rank * config.ngpus_per_node,
                              group=cnt_node_group,
                              tag=(it*dist.get_world_size()+dist.get_rank())*3+2)]
            for op in ops:
                op.wait()

        model.train()
        adjust_learning_rate(optimizer, config.get_learning_rate(it))
        if dist.get_rank() == 0:
            logging.info('=> [Iter:{}]\tTraining starts'.format(it + 1))
            losses_p = AverageMeter('Loss_p', ':4f')
            losses_v = AverageMeter('Loss_v', ':4f')
            progress = ProgressMeter(
                config.train_steps,
                [losses_p, losses_v],
                prefix='[Iter:{}]'.format(it + 1))

        data = data.to(device)
        prob = prob.to(device)
        value = value.to(device)
        for i in range(config.train_steps):
            _data = data[i*config.batch_size_per_gpu:(i+1)*config.batch_size_per_gpu]
            _prob = prob[i*config.batch_size_per_gpu:(i+1)*config.batch_size_per_gpu]
            _value = value[i*config.batch_size_per_gpu:(i+1)*config.batch_size_per_gpu]

            p, v = model(_data, use_softmax=False)
            loss_p = -(_prob * F.log_softmax(p, dim=1)).mean()
            loss_v = ((_value - v) ** 2).mean()
            if dist.get_rank() == 0:
                losses_p.update(loss_p.item(), _data.size(0))
                losses_v.update(loss_v.item(), _data.size(0))

            optimizer.zero_grad()
            (loss_p + loss_v).backward()
            optimizer.step()

            if dist.get_rank() == 0 and (i + 1) % config.print_freq == 0:
                logging.info(progress.to_string(i + 1))

        if dist.get_rank() == 0:
            logging.info('=> [Iter:{}]\tStart saving module and checkpoint'.format(it + 1))
            save_module(it + 1, model.module, config) 
            save_ckpt(it + 1, model.module, config)
            logging.info('=> [Iter:{}]\tDone saving module and checkpoint'.format(it + 1))

        dist.broadcast(acc_tensor, src=0, group=acc_process_group)
        if acc_tensor.item() > best_acc:
            best_acc = acc_tensor.item()
            best_iter = it
        if acc_tensor.item() >= 0.95 or it - best_iter >= 5:
            lo += 1
            hi += 1
            best_acc = 0.0
            if hi > config.nbox:
                return

def save_module(it, model, config):
    # remember model is a distributed model so
    # to trace it we need to use .module to get its original model
    model.eval()
    with torch.no_grad():
        inp = torch.zeros(config.script_batch_size,
                          config.n, config.m, dtype=torch.long).to(config.device)
        smodule = torch.jit.trace(model.forward, inp)
        smodule.save(os.path.join(config.smod_dir, 'iter-{}'.format(it)))

def save_ckpt(it, model, config):
    torch.save({
        'state_dict': model.state_dict(),
    }, os.path.join(config.ckpt_dir, 'iter-{}'.format(it)))

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
