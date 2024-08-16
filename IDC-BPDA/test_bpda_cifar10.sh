#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval_sde_adv_bpda_idc.py --config cifar10-bpda.yml \
--exp ./exp_results \
--i base_alpha0.2 \
--adv_eps 0.031373 \
--adv_batch_size 2 \
--num_sub -1 \
--domain cifar10 \
--diffusion_type idc \
--sample_to_eval \
--resume_model checkpoints/latest_model_400.pth