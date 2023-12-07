import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downblock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(Downblock, self).__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, groups=channels, stride=2, kernel_size=kernel_size, padding=1, bias=False
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.dwconv(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, spatial=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.downop = Downblock(self.expansion * planes, kernel_size=spatial)
        mlp = False
        self.mlp = (
            nn.Sequential(
                nn.Conv2d(planes, planes // 16, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes // 16, planes, kernel_size=1, bias=False),
            )
            if mlp
            else lambda x: x
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        map = self.mlp(self.downop(out))

        # Assuming squares because lazy.
        map = F.interpolate(map, out.shape[-1])
        out = self.shortcut(x) + out * torch.sigmoid(map)
        out = F.relu(out)
        return out


class GENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(GENet, self).__init__()
        self.in_planes = 64
        self.spatial = [8, 4, 2, 1]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, spatial=self.spatial[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, spatial=self.spatial[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, spatial=self.spatial[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, spatial=self.spatial[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, spatial):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, spatial))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def GENet_ResNet50(args):
    return GENet(Bottleneck, [3, 4, 6, 3], args.num_classes)


class GENet_Res50_model(nn.Module):
    def __init__(self, args):
        super(GENet_Res50_model, self).__init__()
        self.model = GENet_ResNet50(args)

    def forward(self, x):
        return self.model(x)
