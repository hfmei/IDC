import torch
from model.BrownianBridge.base.modules.resnet.resnet_model import ResNet50

# def print_model_summary(net):
#     def get_parameter_number(model):
#         total_num = sum(p.numel() for p in model.parameters())
#         trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         return total_num, trainable_num
#
#     total_num, trainable_num = get_parameter_number(net)
#     print("Total Number of parameter: %.2fM" % (total_num / 1e6))
#     print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

x = torch.rand((4, 3, 32, 32))
resnet = ResNet50()
# print_model_summary(resnet)
y = resnet(x)


