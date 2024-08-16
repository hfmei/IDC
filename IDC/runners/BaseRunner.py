import argparse
import datetime
import pdb
import time

import yaml
import os
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

from evaluation.FID import calc_FID
from evaluation.LPIPS import calc_LPIPS
from runners.base.EMA import EMA
from runners.utils import make_save_dirs, make_dir, get_dataset_proto, remove_file

import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist

import torchattacks

class SDE_Adv_Model(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, x):
        self.eval()
        dims = 3 * 32 * 32
        num_class = self.config.model.num_classes
        torch.manual_seed(0)
        initial_tensor = torch.randn(num_class, dims)
        q, _ = torch.qr(initial_tensor.T)
        orthogonal_proto = [q[:, i].view(3, 32, 32) for i in range(num_class)]
        normalized_proto = [(tensor - tensor.min()) / (tensor.max() - tensor.min()) for tensor in orthogonal_proto]
        prototypes = torch.stack(normalized_proto, dim=0).to(x.device)

        prototypes = prototypes.to(self.config.training.device[0])

        x_re = self.model.p_sample_loop(x)
        x_re = x_re.mul_(0.5).add_(0.5).clamp_(0, 1.)
        distence = torch.mean((x_re.unsqueeze(1).repeat(
            1, prototypes.size()[0], 1, 1, 1) - prototypes.unsqueeze(0).repeat(
            x_re.size()[0], 1, 1, 1, 1)).abs(), dim=(-3, -2, -1))
        tao = 0.1

        out = F.softmax(-distence/tao, dim=-1)

        return out


class Torch_Adv_Model(nn.Module):
    def __init__(self, net, config):
        super().__init__()
        self.config = config
        self.net = net
        self.model = SDE_Adv_Model(self.config, self.net)

    def forward(self, train_batch, attack_type):
        (x_cond, x, label) = train_batch
        self.model.eval()
        if attack_type == 'pgd':
            atk = torchattacks.PGD(self.model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
            with torch.enable_grad():
                adv_images = atk(x_cond, label)
        elif attack_type == 'apgd':
            atk = torchattacks.APGD(self.model, eps=8 / 255, steps=10)
            with torch.enable_grad():
                adv_images = atk(x_cond, label)
        elif attack_type == 'fgsm':
            atk = torchattacks.FGSM(self.model, eps=8 / 255)
            with torch.enable_grad():
                adv_images = atk(x_cond, label)
        elif attack_type == 'mifgsm':
            atk = torchattacks.MIFGSM(self.model, eps=8 / 255, alpha=2 / 255, steps=5, decay=1.0)
            with torch.enable_grad():
                adv_images = atk(x_cond, label)
        elif attack_type == 'eotpgd':
            atk = torchattacks.EOTPGD(self.model, eps=8 / 255, alpha=2 / 255, steps=200, eot_iter=20, random_start=True)
            with torch.enable_grad():
                adv_images = atk(x_cond, label)
        elif attack_type == 'cwl2':
            atk = torchattacks.CW(self.model, c=1e-4, kappa=0, steps=1000, lr=0.01)
            with torch.enable_grad():
                adv_images = atk(x_cond, label)
        elif attack_type == 'autoattack':
            atk = torchattacks.AutoAttack(self.model, n_classes=self.config.model.num_classes)
            with torch.enable_grad():
                adv_images = atk(x_cond, label)

        adv_images.to('cpu')
        x.to('cpu')
        adv_batch = (adv_images, x, label)
        return adv_batch

class BaseRunner(ABC):
    def __init__(self, config, logger=None):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file
        self.logger = logger

        # set training params
        self.global_epoch = 0  # global epoch
        if config.args.sample_at_start:
            self.global_step = -1  # global step
        else:
            self.global_step = 0

        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints

        # set log and save destination
        self.config.result = argparse.Namespace()
        self.config.result.image_path, \
        self.config.result.ckpt_path, \
        self.config.result.log_path, \
        self.config.result.sample_path, \
        self.config.result.sample_to_eval_path = make_save_dirs(self.config.args,
                                                                prefix=self.config.data.dataset_name,
                                                                suffix=self.config.model.model_name)

        self.save_config()  # save configuration file
        self.writer = SummaryWriter(self.config.result.log_path)  # initialize SummaryWriter

        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.config)

        self.print_model_summary(self.net)

        # initialize EMA
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.ema.register(self.net)

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        if self.config.training.use_DDP:
            self.net = DDP(self.net, device_ids=[self.config.training.local_rank], output_device=self.config.training.local_rank)
        else:
            self.net = self.net.to(self.config.training.device[0])
        # self.ema.reset_device(self.net)

    # save configuration file
    def save_config(self):
        save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            self.logger.info(f"load model {self.config.model.model_name} from {self.config.model.model_load_path}")
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.net.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.net)

            # load optimizer and scheduler
            if self.config.args.train:
                if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
        }

        if self.config.training.use_DDP:
            model_states['model'] = self.net.module.state_dict()
        else:
            model_states['model'] = self.net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        if self.config.training.use_DDP:
            self.ema.update(self.net.module, with_decay=with_decay)
        else:
            self.ema.update(self.net, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.apply_shadow(self.net.module)
            else:
                self.ema.apply_shadow(self.net)

    def restore_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.restore(self.net.module)
            else:
                self.ema.restore(self.net)

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step, val_dataset):
        self.apply_ema()
        self.net.eval()
        loss, recloss, simloss = self.loss_fn(net=self.net,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step',
                            dataset=val_dataset,)
        if len(self.optimizer) > 1:
            loss, recloss, simloss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step',
                                dataset=val_dataset,)
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch, val_dataset):
        self.apply_ema()
        self.net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss, recloss, simloss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False,
                                dataset=val_dataset)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss, recloss, simloss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False,
                                    dataset=val_dataset)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
        if len(self.optimizer) > 1:
            average_dloss = dloss_sum / step
            self.writer.add_scalar(f'val_dloss_epoch/loss', average_dloss, epoch)
        self.restore_ema()
        return average_loss

    @torch.no_grad()
    def sample_step(self, train_batch, val_batch):
        self.apply_ema()
        self.net.eval()
        sample_path = make_dir(os.path.join(self.config.result.image_path, str(self.global_step)))
        if self.config.training.use_DDP:
            self.sample(self.net.module, train_batch, sample_path, stage='train')
            self.sample(self.net.module, val_batch, sample_path, stage='val')
        else:
            self.sample(self.net, train_batch, sample_path, stage='train')
            self.sample(self.net, val_batch, sample_path, stage='val')
        self.restore_ema()

    @torch.no_grad()
    def calculate_epoch(self, train_loader, val_loader, prototypes):
        self.apply_ema()
        self.net.eval()
        len_acc = len(val_loader)
        pbar = tqdm(val_loader, total=len_acc, smoothing=0.01)
        acc_sample_dist = torch.tensor(0.0, device=self.config.training.device[0])
        count = 0
        for batch in pbar:
            count += 1
            if count > len_acc:
                break
            (x_cond, x, label) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            if self.config.training.use_DDP:
                sample = self.net.module.sample(x_cond, clip_denoised=self.config.testing.clip_denoised).to('cpu')
            else:
                sample = self.net.sample(x_cond, clip_denoised=self.config.testing.clip_denoised).to('cpu')
            if self.config.data.dataset_config.to_normal:
                sample = sample.mul_(0.5).add_(0.5).clamp_(0, 1.)
            for i in range(len(sample)):
                x_sample = sample[i: i+1]
                y_sample_dist = torch.argmin(torch.mean((x_sample.repeat(
                    prototypes.size()[0], 1, 1, 1) - prototypes).abs(), dim=(1, 2, 3)))
                if y_sample_dist.item() == label[i]:
                    acc_sample_dist = acc_sample_dist + 1.0

        self.restore_ema()

        return acc_sample_dist


    # abstract methods
    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, dataset, opt_idx=0, stage='train', write=True):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass


    @abstractmethod
    def sample(self, net, batch, sample_path, stage='train'):
        """
        sample a single batch
        :param net: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass

    @abstractmethod
    def sample_to_eval(self, net, test_loader, sample_path, prototypes, adversary):
        """
        sample among the test dataset to calculate evaluation metrics
        :param net: nn.Module
        :param test_loader: test dataloader
        :param sample_path: path to save samples
        :return:
        """
        pass

    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        """
        additional operations whilst saving checkpoint
        :param net: nn.Module
        :param train_loader: train data loader
        :param val_loader: val data loader
        :param epoch: epoch
        :param step: step
        :return:
        """
        pass

    def train(self):
        self.logger.info(self.__class__.__name__)

        train_dataset, val_dataset = get_dataset_proto(self.config.data)
        train_sampler = None
        val_sampler = None
        if self.config.training.use_DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      num_workers=8,
                                      drop_last=True,
                                      sampler=train_sampler)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    num_workers=8,
                                    drop_last=True,
                                    sampler=val_sampler)
        else:
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      shuffle=self.config.data.train.shuffle,
                                      num_workers=8,
                                      drop_last=True)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    shuffle=self.config.data.val.shuffle,
                                    num_workers=8,
                                    drop_last=True)

        epoch_length = len(train_loader)
        start_epoch = self.global_epoch
        total_steps = self.config.training.n_epochs * len(train_loader)
        self.logger.info(
            f"start training {self.config.model.model_name} on {self.config.data.dataset_name}, {len(train_loader)} iters per epoch")

        try:
            accumulate_grad_batches = self.config.training.accumulate_grad_batches
            for epoch in range(start_epoch, self.config.training.n_epochs):
                # set train
                self.net.train()

                if self.global_step > self.config.training.n_steps:
                    break

                if self.config.training.use_DDP:
                    train_sampler.set_epoch(epoch)
                    val_sampler.set_epoch(epoch)
                self.global_epoch = epoch
                start_time = time.time()
                # pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01)
                with tqdm(train_loader, total=len(train_loader), smoothing=0.01) as pbar:

                    for train_batch in pbar:
                        self.global_step += 1
                        self.net.train()

                        losses = []
                        reclosses = []
                        simlosses = []
                        for i in range(len(self.optimizer)):
                            # pdb.set_trace()
                            loss, recloss, simloss = self.loss_fn(net=self.net,
                                                    batch=train_batch,
                                                    epoch=epoch,
                                                    step=self.global_step,
                                                    opt_idx=i,
                                                    stage='train',
                                                    dataset=train_dataset)

                            loss.backward()
                            if self.global_step % accumulate_grad_batches == 0:
                                self.optimizer[i].step()
                                self.optimizer[i].zero_grad()
                                if self.config.model.auto_lr_scheduler:
                                    if self.scheduler is not None:
                                        self.scheduler[i].step(loss)
                            losses.append(loss.detach().mean())
                            reclosses.append(recloss.detach().mean())
                            simlosses.append(simloss.detach().mean())

                        if self.use_ema and self.global_step % (self.update_ema_interval*accumulate_grad_batches) == 0:
                            self.step_ema()


                        if len(self.optimizer) > 1:
                            pbar.set_description(
                                (
                                    f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                    f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                                    f'recloss-1: {reclosses[0]:.4f} recloss-2: {reclosses[1]:.4f}'
                                    f'clsloss-1: {simlosses[0]:.4f} clsloss-2: {simlosses[1]:.4f}'
                                )
                            )
                        else:
                            lr = self.optimizer[0].param_groups[0]['lr']
                            pbar.set_description(
                                (
                                    f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                    f'iter: {self.global_step} lr: {lr} loss: {losses[0]:.4f} recloss: {reclosses[0]:.4f} clsloss: {simlosses[0]:.4f}'
                                )
                            )
                        if self.global_step % 200 == 0:
                            lr = self.optimizer[0].param_groups[0]['lr']
                            self.logger.info(f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                             f'iter: {self.global_step} lr: {lr} loss: {losses[0]:.4f} recloss: {reclosses[0]:.4f} clsloss: {simlosses[0]:.4f}')

                        with torch.no_grad():
                            if self.global_step % 50 == 0:
                                val_batch = next(iter(val_loader))
                                self.validation_step(val_batch=val_batch, epoch=epoch,
                                                     step=self.global_step, val_dataset = val_dataset)

                            if self.global_step % int(self.config.training.sample_interval * epoch_length) == 0:
                                # val_batch = next(iter(val_loader))
                                # self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_step)

                                if not self.config.training.use_DDP or \
                                        (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                                    val_batch = next(iter(val_loader))
                                    self.sample_step(val_batch=val_batch, train_batch=train_batch)

                                    torch.cuda.empty_cache()

                if not self.config.model.auto_lr_scheduler:
                    if self.scheduler is not None:
                        for i in range(len(self.scheduler)):
                            self.scheduler[i].step()

                end_time = time.time()
                elapsed_rounded = int(round((end_time-start_time)))
                self.logger.info("training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))

                # set eval
                self.net.eval()

                # validation
                if (epoch + 1) % self.config.training.validation_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs:

                    with torch.no_grad():
                        self.logger.info("validating epoch...")
                        average_loss = self.validation_epoch(val_loader, epoch, val_dataset)
                        if not self.config.training.use_DDP or \
                                (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                            self.logger.info("Epoch[{:d}], re_loss:{:.3f}".format(epoch, average_loss))
                        torch.cuda.empty_cache()
                        self.logger.info("validating epoch success")

                        self.logger.info("Calculating accuracy ...")
                        acc_sample_dist = self.calculate_epoch(train_loader,
                                                                         val_loader, val_dataset.prototypes)

                        if self.config.training.use_DDP:
                            dist.all_reduce(acc_sample_dist, op=dist.ReduceOp.SUM)
                        acc_sample_dist = acc_sample_dist / len(val_dataset.dataset.data)

                        if not self.config.training.use_DDP or \
                                (self.config.training.use_DDP and self.config.training.local_rank) == 0:

                            self.logger.info("Epoch[{:d}], accuracy:{:.3f}".format(
                                epoch + 1, acc_sample_dist))

                        torch.cuda.empty_cache()

                # save checkpoint
                if (epoch + 1) % self.config.training.save_interval == 0 or \
                        (epoch + 1) == self.config.training.n_epochs or \
                        self.global_step > self.config.training.n_steps:
                    if not self.config.training.use_DDP or \
                            (self.config.training.use_DDP and self.config.training.local_rank == 0):
                        with torch.no_grad():
                            self.logger.info("saving latest checkpoint...")
                            self.on_save_checkpoint(self.net, train_loader, val_loader, epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')

                            # save latest checkpoint
                            temp = 0
                            while temp < epoch + 1:
                                remove_file(os.path.join(self.config.result.ckpt_path, f'latest_model_{temp}.pth'))
                                remove_file(
                                    os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{temp}.pth'))
                                temp += 1
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_model_{epoch + 1}.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_optim_sche_{epoch + 1}.pth'))
                            # torch.save(model_states,
                            #            os.path.join(self.config.result.ckpt_path,
                            #                         f'last_model.pth'))
                            # torch.save(optimizer_scheduler_states,
                            #            os.path.join(self.config.result.ckpt_path,
                            #                         f'last_optim_sche.pth'))

                            # save top_k checkpoints
                            model_ckpt_name = f'top_model_epoch_{epoch + 1}.pth'
                            optim_sche_ckpt_name = f'top_optim_sche_epoch_{epoch + 1}.pth'

                            if self.config.args.save_top:
                                self.logger.info("save top model start...")
                                top_key = 'top'
                                if top_key not in self.topk_checkpoints:
                                    self.logger.info('top key not in topk_checkpoints')
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                      "acc_sample_dist": acc_sample_dist,
                                                                      'model_ckpt_name': model_ckpt_name,
                                                                      'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    self.logger.info(f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}"
                                          f"acc_sample_dist={acc_sample_dist}")
                                    torch.save(model_states,
                                               os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                    torch.save(optimizer_scheduler_states,
                                               os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                                else:
                                    if acc_sample_dist > self.topk_checkpoints[top_key]["acc_sample_dist"]:
                                        self.logger.info("remove " + self.topk_checkpoints[top_key]["model_ckpt_name"])
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['model_ckpt_name']))
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['optim_sche_ckpt_name']))

                                        self.logger.info(
                                            f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                          "acc_sample_dist": acc_sample_dist,
                                                                          'model_ckpt_name': model_ckpt_name,
                                                                          'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        torch.save(model_states,
                                                   os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                        torch.save(optimizer_scheduler_states,
                                                   os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
        except BaseException as e:
            if not self.config.training.use_DDP or (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                self.logger.info("exception save model start....")
                self.logger.info(self.__class__.__name__)
                model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='exception')
                torch.save(model_states,
                           os.path.join(self.config.result.ckpt_path, f'last_model.pth'))
                torch.save(optimizer_scheduler_states,
                           os.path.join(self.config.result.ckpt_path, f'last_optim_sche.pth'))

                self.logger.info("exception save model success!")

            self.logger.info('str(Exception):\t', str(Exception))
            self.logger.info('str(e):\t\t', str(e))
            self.logger.info('repr(e):\t', repr(e))
            self.logger.info('traceback.print_exc():')
            traceback.print_exc()
            self.logger.info('traceback.format_exc():\n%s' % traceback.format_exc())

    @torch.no_grad()
    def test(self):
        train_dataset, val_dataset = get_dataset_proto(self.config.data)
        # if test_dataset is None:
        test_dataset = val_dataset
        # test_dataset = val_dataset
        if self.config.training.use_DDP:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True)

        if self.use_ema:
            self.apply_ema()

        self.net.eval()

        if not self.config.training.use_DDP:
            adversary = Torch_Adv_Model(self.net, self.config)
        else:
            adversary = Torch_Adv_Model(self.net.module, self.config)


        if self.config.args.sample_to_eval:
            sample_path = self.config.result.sample_to_eval_path
            if self.config.training.use_DDP:
                self.sample_to_eval(self.net.module, test_loader, sample_path, test_dataset.prototypes, adversary)
            else:
                self.sample_to_eval(self.net, test_loader, sample_path, test_dataset.prototypes, adversary)
        else:
            test_iter = iter(test_loader)
            for i in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
                test_batch = next(test_iter)
                sample_path = os.path.join(self.config.result.sample_path, str(i))
                if self.config.training.use_DDP:
                    self.sample(self.net.module, test_batch, sample_path, stage='test')
                else:
                    self.sample(self.net, test_batch, sample_path, stage='test')
