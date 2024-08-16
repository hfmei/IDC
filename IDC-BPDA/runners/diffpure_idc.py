import os
import random
import numpy as np
import torch
import torchvision.utils as tvu
import torchsde
from IDC.runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from IDC.model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from IDC.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from IDC.runners.base.EMA import EMA
from tqdm.autonotebook import tqdm
import torch.nn as nn


class DiffPureIDC(nn.Module):
    def __init__(self, args, config, config_runner, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # initialize model
        self.model, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(config_runner)

        self.print_model_summary(self.model)

        # initialize EMA
        self.use_ema = False if not config_runner.model.__contains__('EMA') else config_runner.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(config_runner.model.EMA.ema_decay)
            self.update_ema_interval = config_runner.model.EMA.update_ema_interval
            self.start_ema_step = config_runner.model.EMA.start_ema_step
            self.ema.register(self.model)

        self.load_model_from_checkpoint(config_runner)


    def print_model_summary(self, model):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(model)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmmodel = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmmodel = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmmodel.apply(weights_init)
        return bbdmmodel

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: model: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        model = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(model, config)
        return model, optimizer, scheduler

    def initialize_optimizer_scheduler(self, model, config):
        optimizer = get_optimizer(config.model.BB.optimizer, model.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
                                                               )
        return [optimizer], [scheduler]

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self, config_runner):
        model_states = None
        if config_runner.model.__contains__('model_load_path') and config_runner.model.model_load_path is not None:
            print(f"load model {config_runner.model.model_name} from {config_runner.model.model_load_path}")
            model_states = torch.load(config_runner.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.model.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.model)

            # load optimizer and scheduler
            if config_runner.args.train:
                if config_runner.model.__contains__(
                        'optim_sche_load_path') and config_runner.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(config_runner.model.optim_sche_load_path,
                                                            map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states


    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False, bs_id=0, tag=None):
        # with torch.no_grad():
        if tag is None:
            tag = 'rnd' + str(random.randint(0, 10000))
        out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)
        # out_dir = './results_test/cifar10'
        assert x_cond.ndim == 4, x_cond.ndim
        x_cond = x_cond.to(self.device)

        if bs_id < 2:
            os.makedirs(out_dir, exist_ok=True)
            tvu.save_image(x_cond, os.path.join(out_dir, f'original_input.png'))


        context = None
        # with torch.no_grad():
        out = self.model.p_sample_loop(x_cond, context, clip_denoised, sample_mid_step)
        out = out.mul_(0.5).add_(0.5).clamp_(0, 1.)

        if bs_id < 2:
            torch.save(out, os.path.join(out_dir, f'samples.pth'))
            tvu.save_image(out, os.path.join(out_dir, f'samples.png'))

        return out