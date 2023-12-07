import os
import mindspore
import sys


class Flatten(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        return x.view(x.shape[0], -1)


class CombConvLayer(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.layer1 = ConvLayer(in_channels, out_channels, kernel)
        self.layer2 = DWConvLayer(out_channels, out_channels, stride=stride)

    def construct(self, x):
        return self.layer2(self.layer1(x))


class DWConvLayer(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        groups = in_channels

        self.dwconv = mindspore.nn.Conv2d(
            in_channels=groups,
            out_channels=groups,
            kernel_size=3,
            stride=stride,
            pad_mode="pad",
            padding=1,
            dilation=1,
            group=groups,
            has_bias=bias,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.norm = mindspore.nn.BatchNorm2d(
            num_features=groups,
            eps=1e-5,
            momentum=0.9,
            affine=True,
            gamma_init="ones",
            beta_init="zeros",
            moving_mean_init="zeros",
            moving_var_init="ones",
            use_batch_statistics=None,
            data_format="NCHW",
        )

    def construct(self, x):
        return self.norm(self.dwconv(x))


class ConvLayer(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.conv = mindspore.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_ch,
            kernel_size=kernel,
            stride=stride,
            pad_mode="pad",
            padding=kernel // 2,
            dilation=1,
            group=groups,
            has_bias=bias,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.norm = mindspore.nn.BatchNorm2d(
            num_features=out_ch,
            eps=1e-5,
            momentum=0.9,
            affine=True,
            gamma_init="ones",
            beta_init="zeros",
            moving_mean_init="zeros",
            moving_var_init="ones",
            use_batch_statistics=None,
            data_format="NCHW",
        )
        self.relu = mindspore.nn.ReLU6()

    def construct(self, x):
        return self.relu(self.norm(self.conv(x)))


class HarDBlock(mindspore.nn.Cell):
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
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = mindspore.nn.CellList(layers_)

    def construct(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = mindspore.ops.cat(tensors=tin, axis=1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = mindspore.ops.cat(tensors=out_, axis=1)
        return out


class HarDNet(mindspore.nn.Cell):
    def __init__(self, depth_wise=True, arch=85, weight_path="", num_classes=1000):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        grmul = 1.7
        drop_rate = 0.1

        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif arch == 39:
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
        self.base = mindspore.nn.CellList([])

        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, bias=False))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))
        self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

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
        self.avgpool = mindspore.ops.ReduceMean(keep_dims=True)
        self.last_layer = mindspore.nn.SequentialCell(
            Flatten(),
            mindspore.nn.Dropout(p=drop_rate),
            mindspore.nn.Dense(
                in_channels=ch,
                out_channels=num_classes,
                weight_init=None,
                bias_init=None,
                has_bias=True,
                activation=None,
            ),
        )

    def construct(self, x):
        for layer in self.base:
            x = layer(x)
        x = self.avgpool(x, (2, 3))
        x = self.last_layer(x)
        return x


class HarDNet_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(HarDNet_model, self).__init__()
        self.model = HarDNet(depth_wise=True, arch=39, num_classes=args.num_classes)

    def construct(self, x):
        return self.model(x)
