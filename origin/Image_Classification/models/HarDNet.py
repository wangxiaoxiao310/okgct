import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CombConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.layer1 = ConvLayer(in_channels, out_channels, kernel)
        self.layer2 = DWConvLayer(out_channels, out_channels, stride=stride)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class DWConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        groups = in_channels

        self.dwconv = nn.Conv2d(groups, groups, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(groups)

    def forward(self, x):
        return self.norm(self.dwconv(x))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.conv = nn.Conv2d(
            in_channels, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2, groups=groups, bias=bias
        )
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2**i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers):
        super().__init__()
        self.keepBase = False
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class HarDNet(nn.Module):
    def __init__(self, depth_wise=True, arch=85, weight_path="", num_classes=1000):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        grmul = 1.7
        drop_rate = 0.1

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, bias=False))
        # Second Layer
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))  # down_sample
        self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            self.base.append(blk)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blks - 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_layer = nn.Sequential(Flatten(), nn.Dropout(drop_rate), nn.Linear(ch, num_classes))

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        x = self.avgpool(x)
        x = self.last_layer(x)
        return x


class HarDNet_model(nn.Module):
    def __init__(self, args):
        super(HarDNet_model, self).__init__()
        self.model = HarDNet(depth_wise=True, arch=39, num_classes=args.num_classes)

    def forward(self, x):
        return self.model(x)
