import torch
import torch.nn.functional as F
import torch.nn as nn

from model.BrownianBridge.base.modules.diffusionmodules.util import timestep_embedding

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes * 4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * 4)
        self.conv4 = nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)
        self.conv5 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)


        # self.bn1 = nn.BatchNorm2d(in_planes)
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes * 4, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        # self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=stride, padding=1, bias=False)
        #
        # self.bn4 = nn.BatchNorm2d(planes * 4)
        # self.conv4 = nn.ConvTranspose2d(planes * 4, planes, kernel_size=2, stride=stride, bias=False)
        # self.bn5 = nn.BatchNorm2d(planes)
        # # self.conv5 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)
        # self.conv5 = nn.ConvTranspose2d(planes, in_planes, kernel_size=2, stride=stride, bias=False)

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(in_planes),
        #         # nn.Conv2d(planes//4, in_planes, kernel_size=1, stride=stride, bias=False),
        #         # nn.BatchNorm2d(in_planes)
        #     )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                planes * 4,
                2 * planes
            ),
        )

    def forward(self, x, emb):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        # # out = F.relu(self.bn3(self.conv3(out)))
        # # out = F.relu(self.bn4(self.conv4(out)))
        # out = self.bn5(self.conv5(out))
        # out += self.shortcut(x)
        # # out = F.relu(out)

        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv4(F.relu(self.bn4(out)))
        out = F.interpolate(out, scale_factor=2, mode="nearest")
        emb_out = self.emb_layers(emb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        out = self.bn5(out) * (1 + scale) + shift
        out = self.conv5(F.relu(out))
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 3
        self.model_channels = 128
        # num_input_channels = 3
        # mean = (0.4914, 0.4822, 0.4465)
        # std = (0.2471, 0.2435, 0.2616)
        # self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        # self.std = torch.tensor(std).view(num_input_channels, 1, 1)

        self.layer1 = self._make_layer(block, self.model_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, self.model_channels, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.model_channels, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.model_channels, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.output_blocks = nn.ModuleList([])
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def forward(self, x, t, T):
    #     # out = (x - self.mean.to(x.device)) / self.std.to(x.device)
    #     # out = F.relu(self.bn1(self.conv1(out)))
    #     out = x
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     # out = F.avg_pool2d(out, 4)
    #     # out = out.view(out.size(0), -1)
    #     # out = self.linear(out)
    #     return out


# def ResNet50():
#     return ResNet(Bottleneck, [4, 4, 4, 4])

class ResNet50(ResNet):
    def __init__(self):
        super(ResNet50, self).__init__(Bottleneck, [4, 4, 4, 4])
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x, t, T):
        quartile_size = T // 4
        layer_id = t // quartile_size
        ### ------ If the number of layers is large, you can open that!------
        # all_id = set(range(4))
        # present_id = set(layer_id.tolist())
        # missing_id = list(all_id - present_id)
        # for id in missing_id:
        #     block = self.get_resnet_layer(id)
        #     for parameter in block.parameters():
        #         parameter.requires_grad = False
        # if len(missing_id) == 0:
        #     for parameter in self.denoise_fn.parameters():
        #         parameter.requires_grad = True

        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        output = x
        for module in self.layer1:
            output = module(output, emb)
        for module in self.layer2:
            output = module(output, emb)
        for module in self.layer3:
            output = module(output, emb)
        for module in self.layer4:
            output = module(output, emb)
        # output = torch.zeros_like(x)
        # for i in range(4):
        #     mask = layer_id == i
        #     if mask.any():
        #         batch_x = x[mask]
        #         batch_emb = emb[mask]
        #         # output_layer = self.get_resnet_layer(i)(batch_x, emb)
        #         for module in self.get_resnet_layer(i):
        #             output_layer = module(batch_x, batch_emb)
        #             output[mask] = output_layer
        return output

    def get_resnet_layer(self, id):
        layer_mapping = {
            0: self.layer1,
            1: self.layer2,
            2: self.layer3,
            3: self.layer4
        }
        return layer_mapping.get(id)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [stride] * (num_blocks - 1)   #TODO: We have change [1] to [stride]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)