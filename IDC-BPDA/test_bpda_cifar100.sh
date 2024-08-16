#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 eval_sde_adv_bpda_idc.py --config cifar100-bpda.yml \
--exp ./exp_results \
--i base_alpha0.2 \
--adv_eps 0.031373 \
--adv_batch_size 2 \
--num_sub -1 \
--domain cifar100 \
--diffusion_type idc \
--sample_to_eval \
--resume_model checkpoints/cifar100/results/last_model.pth