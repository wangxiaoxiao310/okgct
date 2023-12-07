import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, stride=1, padding_mode="reflect", bias=False
            ),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels),  # nnUNet
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(),  # nnUNet
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, stride=1, padding_mode="reflect", bias=False
            ),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels),  # nnUNet
            nn.ReLU(inplace=True)
            # nn.LeakyReLU()  # nnUNet
        )

    def forward(self, x):
        return self.block(x)


# class DownSample(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DownSample, self).__init__()
#         self.block = nn.Sequential(
#             DoubleConv(in_channels, out_channels),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#     def forward(self, x):
#         return self.block(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            # nn.Conv2d(in_channels//2, out_channels, kernel_size=1, stride=1)
            DoubleConv(in_channels // 2, out_channels),
        )

    def forward(self, x):
        return self.block(x)


# 定义UNet模型
class UNet(nn.Module):
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
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(self.channel1, self.channel2)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(self.channel2, self.channel3)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(self.channel3, self.channel4)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_mid = DoubleConv(self.channel4, self.channel5)

        self.up1 = nn.ConvTranspose2d(self.channel5, self.channel4, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(self.channel5, self.channel4)
        self.up2 = nn.ConvTranspose2d(self.channel4, self.channel3, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(self.channel4, self.channel3)
        self.up3 = nn.ConvTranspose2d(self.channel3, self.channel2, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(self.channel3, self.channel2)
        self.up4 = nn.ConvTranspose2d(self.channel2, self.channel1, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(self.channel2, self.channel1)

        self.out_channel = nn.Conv2d(self.channel1, self.out_channels, kernel_size=1, stride=1)

    def forward(self, x):
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
        cat1 = torch.cat([up1, conv4], dim=1)
        up_conv1 = self.up_conv1(cat1)

        up2 = self.up2(up_conv1)
        cat2 = torch.cat([up2, conv3], dim=1)
        up_conv2 = self.up_conv2(cat2)

        up3 = self.up3(up_conv2)
        cat3 = torch.cat([up3, conv2], dim=1)
        up_conv3 = self.up_conv3(cat3)

        up4 = self.up4(up_conv3)
        cat4 = torch.cat([up4, conv1], dim=1)
        up_conv4 = self.up_conv4(cat4)

        out = self.out_channel(up_conv4)
        return out
