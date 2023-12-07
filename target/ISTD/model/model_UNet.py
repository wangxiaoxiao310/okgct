import mindspore


class DoubleConv(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = mindspore.nn.SequentialCell(
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
            mindspore.nn.ReLU(),
            mindspore.nn.Conv2d(
                in_channels=out_channels,
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
            mindspore.nn.ReLU(),
        )

    def construct(self, x):
        return self.block(x)


class UpSample(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.block = mindspore.nn.SequentialCell(
            mindspore.nn.Conv2dTranspose(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
                pad_mode="pad",
                padding=0,
                output_padding=0,
                dilation=1,
                group=1,
                has_bias=True,
                weight_init="normal",
                bias_init="zeros",
            ),
            DoubleConv(in_channels // 2, out_channels),
        )

    def construct(self, x):
        return self.block(x)


class UNet(mindspore.nn.Cell):
    def __init__(self, args):
        super().__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.num_classes
        self.channel_size = 16
        self.channel1 = self.channel_size
        self.channel2 = 2 * self.channel_size
        self.channel3 = 4 * self.channel_size
        self.channel4 = 8 * self.channel_size
        self.channel5 = 16 * self.channel_size

        self.conv1 = DoubleConv(self.in_channels, self.channel1)
        self.down1 = mindspore.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            pad_mode="pad",
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
            data_format="NCHW",
        )
        self.conv2 = DoubleConv(self.channel1, self.channel2)
        self.down2 = mindspore.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            pad_mode="pad",
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
            data_format="NCHW",
        )
        self.conv3 = DoubleConv(self.channel2, self.channel3)
        self.down3 = mindspore.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            pad_mode="pad",
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
            data_format="NCHW",
        )
        self.conv4 = DoubleConv(self.channel3, self.channel4)
        self.down4 = mindspore.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            pad_mode="pad",
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
            data_format="NCHW",
        )

        self.conv_mid = DoubleConv(self.channel4, self.channel5)

        self.up1 = mindspore.nn.Conv2dTranspose(
            in_channels=self.channel5,
            out_channels=self.channel4,
            kernel_size=2,
            stride=2,
            pad_mode="pad",
            padding=0,
            output_padding=0,
            dilation=1,
            group=1,
            has_bias=True,
            weight_init="normal",
            bias_init="zeros",
        )
        self.up_conv1 = DoubleConv(self.channel5, self.channel4)
        self.up2 = mindspore.nn.Conv2dTranspose(
            in_channels=self.channel4,
            out_channels=self.channel3,
            kernel_size=2,
            stride=2,
            pad_mode="pad",
            padding=0,
            output_padding=0,
            dilation=1,
            group=1,
            has_bias=True,
            weight_init="normal",
            bias_init="zeros",
        )
        self.up_conv2 = DoubleConv(self.channel4, self.channel3)
        self.up3 = mindspore.nn.Conv2dTranspose(
            in_channels=self.channel3,
            out_channels=self.channel2,
            kernel_size=2,
            stride=2,
            pad_mode="pad",
            padding=0,
            output_padding=0,
            dilation=1,
            group=1,
            has_bias=True,
            weight_init="normal",
            bias_init="zeros",
        )
        self.up_conv3 = DoubleConv(self.channel3, self.channel2)
        self.up4 = mindspore.nn.Conv2dTranspose(
            in_channels=self.channel2,
            out_channels=self.channel1,
            kernel_size=2,
            stride=2,
            pad_mode="pad",
            padding=0,
            output_padding=0,
            dilation=1,
            group=1,
            has_bias=True,
            weight_init="normal",
            bias_init="zeros",
        )
        self.up_conv4 = DoubleConv(self.channel2, self.channel1)

        self.out_channel = mindspore.nn.Conv2d(
            in_channels=self.channel1,
            out_channels=self.out_channels,
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
        conv1 = self.conv1(x)
        down1 = self.down1(conv1)
        conv2 = self.conv2(down1)
        down2 = self.down2(conv2)
        conv3 = self.conv3(down2)
        down3 = self.down3(conv3)
        conv4 = self.conv4(down3)
        down4 = self.down4(conv4)

        conv_mid = self.conv_mid(down4)

        up1 = self.up1(conv_mid)
        cat1 = mindspore.ops.cat(tensors=[up1, conv4], axis=1)
        up_conv1 = self.up_conv1(cat1)

        up2 = self.up2(up_conv1)
        cat2 = mindspore.ops.cat(tensors=[up2, conv3], axis=1)
        up_conv2 = self.up_conv2(cat2)

        up3 = self.up3(up_conv2)
        cat3 = mindspore.ops.cat(tensors=[up3, conv2], axis=1)
        up_conv3 = self.up_conv3(cat3)

        up4 = self.up4(up_conv3)
        cat4 = mindspore.ops.cat(tensors=[up4, conv1], axis=1)
        up_conv4 = self.up_conv4(cat4)

        out = self.out_channel(up_conv4)
        return out
