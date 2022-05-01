import torch
import torch.nn as nn

__all__ = ['GoogLeNet']


# GoogleNet

class GoogLeNet(nn.Module):

    def __init__(self, num_classes=2):
        super(GoogLeNet, self).__init__()

        self.layer_1 = nn.Sequential(
            ConvBN(1, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBN(64, 64, kernel_size=1),
            ConvBN(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer_2 = nn.Sequential(
            Inception_module(192,  64,  96, 128,  16,  32,  32),
            Inception_module(256, 128, 128, 192,  32,  96,  64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer_3 = nn.Sequential(
            Inception_module(480, 192,  96, 208,  16,  48,  64),
            Inception_module(512, 160, 112, 224,  24,  64,  64),
            Inception_module(512, 128, 128, 256,  24,  64,  64),
            Inception_module(512, 112, 144, 288,  32,  64,  64),
            Inception_module(528, 256, 160, 320,  32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer_4 = nn.Sequential(
            Inception_module(832, 256, 160, 320,  32, 128, 128),
            Inception_module(832, 384, 192, 384,  48, 128, 128),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(1024, num_classes)

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
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out

class Inception_module(nn.Module):

    def __init__(self, in_dim, out_dim,
                 mid_dim_3, out_dim_3,
                 mid_dim_5, out_dim_5, pool):
        super(Inception_module, self).__init__()

        self.conv_1 = ConvBN(in_dim, out_dim, kernel_size=1)

        self.conv_1_3 = nn.Sequential(
            ConvBN(in_dim, mid_dim_3,kernel_size=1),
            ConvBN(mid_dim_3, out_dim_3,kernel_size=3, padding=1)) 

        self.conv_1_5 = nn.Sequential(
            ConvBN(in_dim, mid_dim_5, kernel_size=1),
            ConvBN(mid_dim_5, out_dim_5, kernel_size=5, padding=2)) 

        self.max_3_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBN(in_dim, pool, kernel_size=1))

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_1_3(x)
        out_3 = self.conv_1_5(x)
        out_4 = self.max_3_1(x) 
        out = torch.cat([out_1, out_2, out_3, out_4], dim=1)
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
