import mindspore


class ShuffleBlock(mindspore.nn.Cell):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def construct(self, x):
        N, C, H, W = x.shape
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(mindspore.nn.Cell):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def construct(self, x):
        c = int(x.shape[1] * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(mindspore.nn.Cell):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
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
            num_features=in_channels,
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
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            dilation=1,
            group=in_channels,
            has_bias=False,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.bn2 = mindspore.nn.BatchNorm2d(
            num_features=in_channels,
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
            in_channels=in_channels,
            out_channels=in_channels,
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
            num_features=in_channels,
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
        self.shuffle = ShuffleBlock()

    def construct(self, x):
        x1, x2 = self.split(x)
        out = mindspore.ops.ReLU()(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = mindspore.ops.ReLU()(self.bn3(self.conv3(out)))
        out = mindspore.ops.cat(tensors=[x1, out], axis=1)
        out = self.shuffle(out)
        return out


class DownBlock(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            pad_mode="pad",
            padding=1,
            dilation=1,
            group=in_channels,
            has_bias=False,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.bn1 = mindspore.nn.BatchNorm2d(
            num_features=in_channels,
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
            in_channels=in_channels,
            out_channels=mid_channels,
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
            num_features=mid_channels,
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
            in_channels=in_channels,
            out_channels=mid_channels,
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
            num_features=mid_channels,
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
        self.conv4 = mindspore.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            pad_mode="pad",
            padding=1,
            dilation=1,
            group=mid_channels,
            has_bias=False,
            weight_init=None,
            bias_init=None,
            data_format="NCHW",
        )
        self.bn4 = mindspore.nn.BatchNorm2d(
            num_features=mid_channels,
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
        self.conv5 = mindspore.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
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
        self.bn5 = mindspore.nn.BatchNorm2d(
            num_features=mid_channels,
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

        self.shuffle = ShuffleBlock()

    def construct(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = mindspore.ops.ReLU()(self.bn2(self.conv2(out1)))
        out2 = mindspore.ops.ReLU()(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = mindspore.ops.ReLU()(self.bn5(self.conv5(out2)))
        out = mindspore.ops.cat(tensors=[out1, out2], axis=1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(mindspore.nn.Cell):
    def __init__(self, net_size, num_classes):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]["out_channels"]
        num_blocks = configs[net_size]["num_blocks"]

        self.conv1 = mindspore.nn.Conv2d(
            in_channels=3,
            out_channels=24,
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
            num_features=24,
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
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = mindspore.nn.Conv2d(
            in_channels=out_channels[2],
            out_channels=out_channels[3],
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
            num_features=out_channels[3],
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
            in_channels=out_channels[3],
            out_channels=num_classes,
            weight_init=None,
            bias_init=None,
            has_bias=True,
            activation=None,
        )

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        out = mindspore.ops.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = mindspore.ops.ReLU()(self.bn2(self.conv2(out)))
        out = self.avgpool(out, (2, 3))
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


configs = {
    0.5: {"out_channels": (48, 96, 192, 1024), "num_blocks": (3, 7, 3)},
    1: {"out_channels": (116, 232, 464, 1024), "num_blocks": (3, 7, 3)},
    1.5: {"out_channels": (176, 352, 704, 1024), "num_blocks": (3, 7, 3)},
    2: {"out_channels": (224, 488, 976, 2048), "num_blocks": (3, 7, 3)},
}


class ShuffleNetV2_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(ShuffleNetV2_model, self).__init__()
        self.model = ShuffleNetV2(0.5, args.num_classes)

    def construct(self, x):
        return self.model(x)
