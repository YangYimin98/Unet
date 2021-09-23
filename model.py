import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


# building the Conv part
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of Unet
        for feature in features:
            self.downs.append(Conv(in_channels, feature))
            in_channels = feature

        # Up part of Unet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                        feature * 2, feature, kernel_size=2, stride=2,
                    ))
            self.ups.append(Conv(feature * 2, feature))

        # bottleneck part of Unet
        self.bottleneck = Conv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        res_connections = []
        for down in self.downs:
            x = down(x)
            res_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        res_connections = res_connections[:: -1]

        # when you want to add the res, u only add in conv layer, u need skip sample layer
        for ind in range(0, len(self.ups), 2):
            x = self.ups[ind](x)
            res_connection = res_connections[ind // 2]

            # deal with the irregular input like 81 * 81, which will output 81 * 81
            if x.shape != res_connection.shape:
                x = tf.resize(x, size=res_connection.shape[2:])  # 2: skip batch size and num of channels
            concat_res = torch.cat((res_connection, x), dim=1)
            x = self.ups[ind + 1](concat_res)
        return self.final_conv(x)


def test():
    x = torch.randn(3, 1, 161, 161)
    model = Unet(in_channels=1, out_channels=1)
    pred = model(x)
    print(x.shape)
    print(pred.shape)
    assert x.shape == pred.shape


if __name__ == '__main__':
    test()
