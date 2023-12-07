import math

import mindspore


class Bottleneck(mindspore.nn.Cell):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = mindspore.nn.BatchNorm2d(
            num_features=in_planes,
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
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=in_planes,
            out_channels=4 * growth_rate,
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
            num_features=4 * growth_rate,
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
            in_channels=4 * growth_rate,
            out_channels=growth_rate,
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

    def construct(self, x):
        out = self.conv1(mindspore.ops.ReLU()(self.bn1(x)))
        out = self.conv2(mindspore.ops.ReLU()(self.bn2(out)))
        out = mindspore.ops.cat(tensors=[out, x], axis=1)
        return out


class Transition(mindspore.nn.Cell):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = mindspore.nn.BatchNorm2d(
            num_features=in_planes,
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
        self.conv = mindspore.nn.Conv2d(
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
        )

    def construct(self, x):
        out = self.conv(mindspore.ops.ReLU()(self.bn(x)))
        out = mindspore.ops.AvgPool(kernel_size=2, strides=2)(out)
        return out


class DenseNet(mindspore.nn.Cell):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=3,
            out_channels=num_planes,
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

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = mindspore.nn.BatchNorm2d(
            num_features=num_planes,
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
        self.linear = mindspore.nn.Dense(
            in_channels=num_planes,
            out_channels=num_classes,
            weight_init=None,
            bias_init=None,
            has_bias=True,
            activation=None,
        )

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = mindspore.ops.AvgPool(kernel_size=4, strides=4)(mindspore.ops.ReLU()(self.bn(out)))
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def Densenet121_cifar(args):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=args.num_classes)


class DenseNet121_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(DenseNet121_model, self).__init__()
        self.model = Densenet121_cifar(args)

    def construct(self, x):
        return self.model(x)
