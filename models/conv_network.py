import torch
import torch.nn as nn

__all__ = ['ConvNet']


# VGG model class

class ConvNet(nn.Module):

    def __init__(self, base_dim=64, num_classes=2, kernel_size=3, dilation=1):
        super().__init__()

        padding = (kernel_size + 2 * dilation - 1) // 2

        self.layer_1 = nn.Sequential(
            ConvBN(1, base_dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ConvBN(base_dim, base_dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_2 = nn.Sequential(
            ConvBN(base_dim, base_dim * 2, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ConvBN(base_dim * 2, base_dim * 2, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_3 = nn.Sequential(
            ConvBN(base_dim * 2, base_dim * 4, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ConvBN(base_dim * 4, base_dim * 4, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ConvBN(base_dim * 4, base_dim * 4, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_4 = nn.Sequential(
            ConvBN(base_dim * 4, base_dim * 8, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ConvBN(base_dim * 8, base_dim * 8, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ConvBN(base_dim * 8, base_dim * 8, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_5 = nn.Sequential(
            ConvBN(base_dim * 8, base_dim * 8, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ConvBN(base_dim * 8, base_dim * 8, kernel_size=kernel_size, padding=padding, dilation=dilation),
            ConvBN(base_dim * 8, base_dim * 8, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.Conv2d(base_dim * 8, base_dim * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(base_dim * 16, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avg_pool(out).view(batch_size, -1)
        out = self.fc_layer(out)
        return out


class ConvBN(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, bias=False, **kwargs),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


if __name__ == '__main__':
    model = ConvNet()
