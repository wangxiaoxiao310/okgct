import mindspore


class BasicBlock(mindspore.nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
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

        self.shortcut = mindspore.nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = mindspore.nn.SequentialCell(
                mindspore.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
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
                    num_features=self.expansion * planes,
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
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = mindspore.ops.ReLU()(out)
        return out


class Bottleneck(mindspore.nn.Cell):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
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
            group=1,
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
            out_channels=self.expansion * planes,
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
            num_features=self.expansion * planes,
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
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = mindspore.nn.SequentialCell(
                mindspore.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
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
                    num_features=self.expansion * planes,
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
        out += self.shortcut(x)
        out = mindspore.ops.ReLU()(out)
        return out


class ResNet(mindspore.nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = mindspore.nn.Conv2d(
            in_channels=3,
            out_channels=64,
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
            num_features=64,
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
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = mindspore.ops.ReduceMean(keep_dims=True)
        self.linear = mindspore.nn.Dense(
            in_channels=512 * block.expansion,
            out_channels=num_classes,
            weight_init=None,
            bias_init=None,
            has_bias=True,
            activation=None,
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        out = mindspore.ops.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out, (2, 3))
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


def ResNet18(args):
    return ResNet(BasicBlock, [2, 2, 2, 2], args.num_classes)


def ResNet34(args):
    return ResNet(BasicBlock, [3, 4, 6, 3], args.num_classes)


def ResNet50(args):
    return ResNet(Bottleneck, [3, 4, 6, 3], args.num_classes)


def ResNet101(args):
    return ResNet(Bottleneck, [3, 4, 23, 3], args.num_classes)


def ResNet152(args):
    return ResNet(Bottleneck, [3, 8, 36, 3], args.num_classes)


class Resnet50_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(Resnet50_model, self).__init__()
        self.model = ResNet50(args)

    def construct(self, x):
        return self.model(x)
