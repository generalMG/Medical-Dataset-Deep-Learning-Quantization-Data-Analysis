import torch.nn as nn

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class ResNet(nn.Module):

    def __init__(self, block, num_layer, num_classes=2, in_dim=1, base_dim=64):
        super().__init__()

        modules = [ConvBN(in_dim, base_dim, kernel_size=7, padding=3, stride=2)]
        modules += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        if block is Basic:
            modules += [Basic(base_dim, base_dim) 
                    for _ in range(num_layer[0])]
            
            modules += [block(base_dim, base_dim*2, down=True)]
            modules += [block(base_dim*2, base_dim*2) 
                    for _ in range(num_layer[1]-1)]

            modules += [block(base_dim*2, base_dim*4, down=True)]
            modules += [block(base_dim*4, base_dim*4) 
                    for _ in range(num_layer[2]-1)]

            modules += [block(base_dim*4, base_dim*8, down=True)]
            modules += [block(base_dim*8, base_dim*8) 
                    for _ in range(num_layer[3]-1)]

            last_features = base_dim*8

        elif block is Bottleneck:
            modules += [block(base_dim, base_dim, base_dim*4)]
            modules += [block(base_dim*4, base_dim, base_dim*4) 
                    for _ in range(num_layer[0]-1)]

            modules += [block(base_dim*4, base_dim*2, base_dim*8, down=True)]
            modules += [block(base_dim*8, base_dim*2, base_dim*8) 
                    for _ in range(num_layer[1]-1)]

            modules += [block(base_dim*8, base_dim*4, base_dim*16, down=True)]
            modules += [block(base_dim*16, base_dim*4, base_dim*16) 
                    for _ in range(num_layer[2]-1)]

            modules += [block(base_dim*16, base_dim*8, base_dim*32, down=True)]
            modules += [block(base_dim*32, base_dim*8, base_dim*32) 
                    for _ in range(num_layer[3]-1)]

            last_features = base_dim*32

        self.layer = nn.Sequential(*modules)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(last_features, num_classes)

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

        out = self.layer(x)
        out = self.avg_pool(out)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)

        return out

class Basic(nn.Module):

    def __init__(self, in_dim, out_dim, down=False):
        super().__init__()
        stride = 2 if down else 1

        self.down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_dim),
            ) if down else None

        self.layer = nn.Sequential(
            ConvBN(in_dim, out_dim, kernel_size=3, padding=1, stride=stride),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer(x)
        if self.down:
            x = self.down(x)
        return self.relu(out + x)

class Bottleneck(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, down=False):
        super().__init__()
        stride = 2 if down else 1

        self.down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_dim),
            ) if down else None
        
        self.dim_equalizer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_dim),
            ) if in_dim != out_dim and not down else None

        self.layer = nn.Sequential(
            ConvBN(in_dim, mid_dim, kernel_size=1, stride=stride),
            ConvBN(mid_dim, mid_dim, kernel_size=3, padding=1),
            nn.Conv2d(mid_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layer(x)
        if self.down:
            x = self.down(x)
        if self.dim_equalizer:
            x = self.dim_equalizer(x)
        return self.relu(out + x)

class ConvBN(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, bias=False, **kwargs),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
            return self.layer(x)

def resnet18(**kwargs):
    return ResNet(Basic, num_layer=[2,2,2,2], **kwargs)

def resnet34(**kwargs):
    return ResNet(Basic, num_layer=[3,4,6,3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, num_layer=[3,4,6,3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, num_layer=[3,4,23,3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, num_layer=[3,8,36,3], **kwargs)
