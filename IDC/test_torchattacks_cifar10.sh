#!/bin/bash
for i in $(seq 1 5)
do
  python main.py --config configs/Template-IDC-cifar10-attack-$i.yaml \
  --sample_to_eval \
  --resume_model results/cifar10/BrownianBridge/checkpoint/latest_model_400.pth \
  --gpu_ids 0,1,2,3
done