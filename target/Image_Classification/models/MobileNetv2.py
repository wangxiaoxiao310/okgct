import mindspore


class Block(mindspore.nn.Cell):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
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
            num_features=planes,
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
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            pad_mode="pad",
            padding=1,
            dilation=1,
            group=planes,
            has_bias=False,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.bn2 = mindspore.nn.BatchNorm2d(
            num_features=planes,
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
        self.conv3 = mindspore.nn.Conv2d(
            in_channels=planes,
            out_channels=out_planes,
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
            num_features=out_planes,
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

        self.shortcut = mindspore.nn.SequentialCell()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = mindspore.nn.SequentialCell(
                mindspore.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
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
                ),
                mindspore.nn.BatchNorm2d(
                    num_features=out_planes,
                    eps=1e-5,
                    momentum=0.9,
                    affine=True,
                    gamma_init="ones",
                    beta_init="zeros",
                    moving_mean_init="zeros",
                    moving_var_init="ones",
                    use_batch_statistics=None,
                    data_format="NCHW",
                ),
            )

    def construct(self, x):
        out = mindspore.ops.ReLU()(self.bn1(self.conv1(x)))
        out = mindspore.ops.ReLU()(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(mindspore.nn.Cell):
    cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
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
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = mindspore.nn.Conv2d(
            in_channels=320,
            out_channels=1280,
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
        self.bn2 = mindspore.nn.BatchNorm2d(
            num_features=1280,
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
        self.avgpool = mindspore.ops.ReduceMean(keep_dims=True)
        self.linear = mindspore.nn.Dense(
            in_channels=1280, out_channels=num_classes, weight_init=None, bias_init=None, has_bias=True, activation=None
        )

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        out = mindspore.ops.ReLU()(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = mindspore.ops.ReLU()(self.bn2(self.conv2(out)))
        out = self.avgpool(out, (2, 3))
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


class MobileNetv2_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(MobileNetv2_model, self).__init__()
        self.model = MobileNetV2(args.num_classes)

    def construct(self, x):
        return self.model(x)
