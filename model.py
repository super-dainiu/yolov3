import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.batch_norm = batch_norm

    def forward(self, x):
        if self.batch_norm:
            return self.relu(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                )
            ]

        self.relu = nn.LeakyReLU(0.1)
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x) + x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=3, padding=1),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1),
            CNNBlock(in_channels, num_anchors * (num_classes + 5), batch_norm=False, kernel_size=1)
        )
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        return (self.pred(x)
                .reshape(x.shape[0], self.num_anchors, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
                )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.darknet, out_channels = self._create_darknet()
        self.predictor_1 = self._create_predictor(out_channels)
        self.predictor_2 = self._create_predictor(out_channels // 2)
        self.predictor_3 = self._create_predictor(out_channels // 4)
        self.upsampler_1 = nn.Upsample(scale_factor=2)
        self.linear_1 = CNNBlock(out_channels // 2 * 3, out_channels // 2, kernel_size=1)
        self.upsampler_2 = nn.Upsample(scale_factor=2)
        self.linear_2 = CNNBlock(out_channels // 4 * 3, out_channels // 4, kernel_size=1)

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.darknet:
            x = layer(x)
            if isinstance(layer, ResidualBlock):
                if layer.num_repeats == 8:
                    route_connections.append(x)

        outputs.append(self.predictor_1(x))

        x = self.upsampler_1(x)
        x = self.linear_1(torch.cat([x, route_connections.pop()], dim=1))
        outputs.append(self.predictor_2(x))

        x = self.upsampler_2(x)
        x = self.linear_2(torch.cat([x, route_connections.pop()], dim=1))
        outputs.append(self.predictor_3(x))
        return outputs

    def _create_darknet(self):
        return nn.ModuleList([
            CNNBlock(self.in_channels, 32, kernel_size=3, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8),
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8),
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),
        ]), 1024

    def _create_predictor(self, in_channels):
        return nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=1),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1),
            CNNBlock(in_channels, in_channels // 2, kernel_size=1),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1),
            ScalePrediction(in_channels, num_classes=self.num_classes)
        )


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Successfully build model YOLOv3-VOC")
