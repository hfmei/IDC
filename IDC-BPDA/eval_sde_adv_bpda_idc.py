# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import logging
import yaml
import os
import time

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpda_eot.bpda_eot_attack import BPDA_EOT_Attack

import utils
from utils import str2bool, get_accuracy, get_image_classifier, load_data

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_sde import RevGuidedDiffusion
from runners.diffpure_idc import DiffPureIDC
import torchvision.utils as tvu

import uuid


class ResNet_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        # image classifier
        self.resnet = get_image_classifier(args.classifier_name).to(config.device)

    def purify(self, x):
        return x

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            out = self.purify(x)
        elif mode == 'classify':
            out = self.resnet(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            x = self.purify(x)
            out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        # image classifier
        # self.resnet = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        elif args.diffusion_type == 'idc':
            self.runner = DiffPureIDC(args, config.idc.model, config.idc, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
    # and you may want to change the freq)
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def apply_ema(self):
        if self.runner.use_ema:
            if torch.cuda.device_count() > 1:
                self.runner.ema.apply_shadow(self.runner.model)
            else:
                self.runner.ema.apply_shadow(self.runner.model)

    def purify(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        if self.args.diffusion_type == 'idc':
            if self.runner.use_ema:
                self.apply_ema()

            # if self.to_normal:

            x_re = self.runner.sample(x, clip_denoised=True, bs_id=counter, tag=self.tag)
            minutes, seconds = divmod(time.time() - start_time, 60)

            if 'imagenet' in self.args.domain:
                x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)
            # else:
            #     x_re = torchvision.transforms.Resize(32)(x_re)

            if counter % 5 == 0:
                print(f'x shape (before diffusion models): {x.shape}')
                print(f'x shape (before classifier): {x_re.shape}')
                print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

            # out = x_re
            dims = 3 * 32 * 32
            num_class = self.config.idc.model.num_classes
            torch.manual_seed(0)
            initial_tensor = torch.randn(num_class, dims)
            q, _ = torch.qr(initial_tensor.T)
            orthogonal_proto = [q[:, i].view(3, 32, 32) for i in range(num_class)]
            normalized_proto = [(tensor - tensor.min()) / (tensor.max() - tensor.min()) for tensor in orthogonal_proto]
            prototypes = torch.stack(normalized_proto, dim=0).to(x.device)

            distence = torch.mean((x_re.unsqueeze(1).repeat(
                1, prototypes.size()[0], 1, 1, 1) - prototypes.unsqueeze(0).repeat(
                x_re.size()[0], 1, 1, 1, 1)).abs(), dim=(-3, -2, -1))
            tao = 0.1
            out = F.softmax(-distence / tao, dim=-1)
        else:
            x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
            # x_re = self.runner.sample((x - 0.5) * 2, clip_denoised=False, bs_id=counter, tag=self.tag)
            minutes, seconds = divmod(time.time() - start_time, 60)

            if 'imagenet' in self.args.domain:
                x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

            if counter % 5 == 0:
                print(f'x shape (before diffusion models): {x.shape}')
                print(f'x shape (before classifier): {x_re.shape}')
                print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

            out = (x_re + 1) * 0.5


        self.counter += 1

        return out

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            # out = self.purify(x)
            out = x
        elif mode == 'classify':
            out = self.purify(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            out = self.purify(x)
            # out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


def eval_bpda(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module

    x_val, y_val = x_val.to(config.device), y_val.to(config.device)

    # # ------------------ apply the attack to resnet ------------------
    # print(f'apply the bpda attack to resnet...')
    # resnet_bpda = ResNet_Adv_Model(args, config)
    # if ngpus > 1:
    #     resnet_bpda = torch.nn.DataParallel(resnet_bpda)
    #
    # start_time = time.time()
    # init_acc = get_accuracy(resnet_bpda, x_val, y_val, bs=adv_batch_size)
    # print('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))
    #
    # adversary_resnet = BPDA_EOT_Attack(resnet_bpda, adv_eps=args.adv_eps, eot_defense_reps=args.eot_defense_reps,
    #                                    eot_attack_reps=args.eot_attack_reps)
    #
    # start_time = time.time()
    # class_batch, ims_adv_batch = adversary_resnet.attack_all(x_val, y_val, batch_size=adv_batch_size)
    #
    # # class_batch, ims_adv_batch, lbls_adv_batch = adversary_resnet.attack_all_labels(x_val, y_val, batch_size=adv_batch_size)
    # # for img_id in range(len(ims_adv_batch)):
    # #     random_id = str(uuid.uuid4()) + '_' + str(np.array(lbls_adv_batch[img_id], dtype=int))
    # #     os.makedirs(os.path.join(log_dir, 'val/A'), exist_ok=True)
    # #     os.makedirs(os.path.join(log_dir, 'val/B'), exist_ok=True)
    # #     tvu.save_image(ims_adv_batch[img_id], os.path.join(os.path.join(log_dir, 'val/A'), f'{random_id}.png'))
    # #     tvu.save_image(x_val[img_id], os.path.join(os.path.join(log_dir, 'val/B'), f'{random_id}.png'))
    #
    # init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
    # robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]
    #
    # print('init acc: {:.2%}, robust acc: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, robust_acc, time.time() - start_time))
    #
    # print(f'x_adv_resnet shape: {ims_adv_batch.shape}')
    # torch.save([ims_adv_batch, y_val], f'{log_dir}/x_adv_resnet_sd{args.seed}.pt')

    # ------------------ apply the attack to sde_adv ------------------
    print(f'apply the bpda attack to sde_adv...')

    start_time = time.time()
    model_.reset_counter()
    model_.set_tag('no_adv')
    # init_acc = get_accuracy(model, x_val, y_val, bs=adv_batch_size)
    # print('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))

    adversary_sde = BPDA_EOT_Attack(model, adv_eps=args.adv_eps, eot_defense_reps=args.eot_defense_reps,
                                    eot_attack_reps=args.eot_attack_reps)
    #
    start_time = time.time()
    model_.reset_counter()
    model_.set_tag()
    class_batch, ims_adv_batch = adversary_sde.attack_all(x_val, y_val, batch_size=adv_batch_size)
    init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
    robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]

    print('init acc: {:.2%}, robust acc: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, robust_acc, time.time() - start_time))

    print(f'x_adv_sde shape: {ims_adv_batch.shape}')
    torch.save([ims_adv_batch, y_val], f'{log_dir}/x_adv_sde_sd{args.seed}.pt')


def robustness_eval(args, config):
    middle_name = '_'.join([args.diffusion_type, 'bpda'])
    log_dir = os.path.join(args.image_folder, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(config.device)

    # load data
    x_val, y_val = load_data(args, adv_batch_size)

    # eval classifier and sde_adv against bpda attack
    eval_bpda(args, config, model, x_val, y_val, adv_batch_size, log_dir)

    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde, celebahq-ddpm]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    parser.add_argument('--eot_defense_reps', type=int, default=150)
    parser.add_argument('--eot_attack_reps', type=int, default=15)

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)

    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)

    # BBDM
    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='sample at start(for debug)')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")
    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')

    # dataset
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='optimizer checkpoint')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        # config = yaml.safe_load(f)
        config = yaml.load(f, Loader=yaml.FullLoader)
    new_config = utils.dict2namespace(config)

    new_config.idc.args = args

    if args.resume_model is not None:
        new_config.idc.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        new_config.idc.model.optim_sche_load_path = args.resume_optim

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device
    new_config.idc.training.device = [device]
    new_config.idc.training.use_DDP = False

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    robustness_eval(args, config)


