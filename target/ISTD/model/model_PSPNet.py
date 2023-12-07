import mindspore


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return mindspore.nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        pad_mode="pad",
        padding=dilation,
        dilation=dilation,
        group=1,
        has_bias=False,
        weight_init=None,
        bias_init=None,
        data_format="NCHW",
    )


class BasicBlock(mindspore.nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
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
        self.relu = mindspore.nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
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
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(mindspore.nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=inplanes,
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
            padding=dilation,
            dilation=dilation,
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
            out_channels=planes * 4,
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
            num_features=planes * 4,
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
        self.relu = mindspore.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(mindspore.nn.Cell):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = mindspore.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            pad_mode="pad",
            padding=3,
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
        self.relu = mindspore.nn.ReLU()
        self.maxpool = mindspore.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            pad_mode="pad",
            padding=1,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
            data_format="NCHW",
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = mindspore.nn.SequentialCell(
                mindspore.nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
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
                    num_features=planes * block.expansion,
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

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return mindspore.nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


class PSPModule(mindspore.nn.Cell):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = mindspore.nn.CellList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = mindspore.nn.Conv2d(
            in_channels=features * (len(sizes) + 1),
            out_channels=out_features,
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
        self.relu = mindspore.nn.ReLU()

    def _make_stage(self, features, size):
        prior = mindspore.nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = mindspore.nn.Conv2d(
            in_channels=features,
            out_channels=features,
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
        return mindspore.nn.SequentialCell(prior, conv)

    def construct(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        priors = [
            mindspore.ops.interpolate(
                input=stage(feats),
                size=(h, w),
                scale_factor=None,
                mode="bilinear",
                align_corners=None,
                recompute_scale_factor=None,
            )
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(mindspore.ops.cat(tensors=priors, axis=1))
        return self.relu(bottle)


class PSPUpsample(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = mindspore.nn.SequentialCell(
            mindspore.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
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
            ),
            mindspore.nn.BatchNorm2d(
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
            ),
            mindspore.nn.PReLU(),
        )

    def construct(self, x):
        h, w = 2 * x.shape[2], 2 * x.shape[3]
        p = mindspore.ops.interpolate(
            input=x, size=(h, w), scale_factor=None, mode="bilinear", align_corners=None, recompute_scale_factor=None
        )
        return self.conv(p)


class PSPNet(mindspore.nn.Cell):
    def __init__(self, args):
        super().__init__()
        num_classes = args.num_classes
        sizes = (1, 2, 3, 6)
        psp_size = 512
        backend = "resnet18"

        self.feats = resnet18()
        self.psp = PSPModule(psp_size, 256, sizes)
        self.drop_1 = mindspore.nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(256, 64)
        self.up_2 = PSPUpsample(64, 16)
        self.up_3 = PSPUpsample(16, 16)

        self.drop_2 = mindspore.nn.Dropout2d(p=0.15)
        self.final = mindspore.nn.Conv2d(
            in_channels=16,
            out_channels=num_classes,
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
        f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p)
