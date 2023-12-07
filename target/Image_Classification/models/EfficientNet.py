import mindspore


def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = mindspore.numpy.empty(shape=[x.shape[0], 1, 1, 1], dtype=mindspore.float32)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(mindspore.nn.Cell):
    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = mindspore.nn.Conv2d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            padding=0,
            dilation=1,
            group=1,
            has_bias=True,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.se2 = mindspore.nn.Conv2d(
            in_channels=se_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            padding=0,
            dilation=1,
            group=1,
            has_bias=True,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.avgpool = mindspore.ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        out = self.avgpool(x, (2, 3))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio=1, se_ratio=0.0, drop_rate=0.0):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        channels = expand_ratio * in_channels
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            padding=0,
            dilation=1,
            group=1,
            has_bias=False,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.bn1 = mindspore.nn.BatchNorm2d(
            num_features=channels,
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

        self.conv2 = mindspore.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=(1 if kernel_size == 3 else 2),
            dilation=1,
            group=channels,
            has_bias=False,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.bn2 = mindspore.nn.BatchNorm2d(
            num_features=channels,
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

        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        self.conv3 = mindspore.nn.Conv2d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            padding=0,
            dilation=1,
            group=1,
            has_bias=False,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.bn3 = mindspore.nn.BatchNorm2d(
            num_features=out_channels,
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

        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def construct(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(mindspore.nn.Cell):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            dilation=1,
            group=1,
            has_bias=False,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.bn1 = mindspore.nn.BatchNorm2d(
            num_features=32,
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
        self.layers = self._make_layers(in_channels=32)
        self.avgpool = mindspore.ops.ReduceMean(keep_dims=True)
        self.linear = mindspore.nn.Dense(
            in_channels=cfg["out_channels"][-1],
            out_channels=num_classes,
            weight_init=None,
            bias_init=None,
            has_bias=True,
            activation=None,
        )

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ["expansion", "out_channels", "num_blocks", "kernel_size", "stride"]]
        b = 0
        blocks = sum(self.cfg["num_blocks"])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg["drop_connect_rate"] * b / blocks
                layers.append(
                    Block(in_channels, out_channels, kernel_size, stride, expansion, se_ratio=0.25, drop_rate=drop_rate)
                )
                in_channels = out_channels
        return mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avgpool(out, (2, 3))
        out = out.view(out.shape[0], -1)
        dropout_rate = self.cfg["dropout_rate"]
        if self.training and dropout_rate > 0:
            out = mindspore.ops.dropout(input=out, p=dropout_rate, training=True, seed=None)
        out = self.linear(out)
        return out


def EfficientNetB3(num_classes):
    cfg = {
        "num_blocks": [1, 2, 2, 3, 3, 4, 1],
        "expansion": [1, 6, 6, 6, 6, 6, 6],
        "out_channels": [16, 24, 40, 80, 112, 192, 320],
        "kernel_size": [3, 3, 5, 3, 5, 5, 3],
        "stride": [1, 2, 2, 2, 1, 2, 1],
        "dropout_rate": 0.2,
        "drop_connect_rate": 0.2,
    }
    return EfficientNet(cfg, num_classes)


class Efficientnet_b3_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(Efficientnet_b3_model, self).__init__()
        self.model = EfficientNetB3(args.num_classes)

    def construct(self, x):
        return self.model(x)
