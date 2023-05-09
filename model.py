import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        size = x.shape[0]
        x = self.pool1(torch.sigmoid(self.conv1(x)))
        x = self.pool2(torch.sigmoid(self.conv2(x)))
        x = x.view(size, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super(AlexNet, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.features3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for i in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class DenseNetBlock(nn.Module):
    def __init__(self, input_features, growth_rate):
        super(DenseNetBlock, self).__init__()
        # BN-ReLU-Conv
        self.batch1 = nn.BatchNorm2d(input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(input_features, 4*growth_rate, kernel_size=1, stride=1, padding=0, bias=False)

        self.batch2 = nn.BatchNorm2d(4*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.batch1(x)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.batch2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = torch.cat([out, x], dim=1)

        return out


class TransitionLayer(nn.Module):
    def __init__(self, output_features):
        super(TransitionLayer, self).__init__()
        self.batch = nn.BatchNorm2d(output_features*2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv = nn.Conv2d(output_features*2, output_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.batch(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv(out)
        out = self.avgpool(out)

        return out



class DenseNet(nn.Module):
    def __init__(self, growth_rate, input_features, num_layers, num_classes):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        self.num_layers = num_layers

        # 输入部分
        self.Inputs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=input_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # We refer the DenseNet with θ <1 as DenseNet-C, and we set θ = 0.5
        self.densenetblock1 = self._make_dense_block(num_layers[0], input_features, 0)
        input_features += num_layers[0] * self.growth_rate
        self.transition1 = self._make_transition_layer(input_features // 2, 0)

        input_features = input_features // 2
        self.densenetblock2 = self._make_dense_block(num_layers[1], input_features, 1)
        input_features += num_layers[1] * self.growth_rate
        self.transition2 = self._make_transition_layer(input_features // 2, 1)

        input_features = input_features // 2
        self.densenetblock3 = self._make_dense_block(num_layers[2], input_features, 2)
        input_features += num_layers[2] * self.growth_rate
        self.transition3 = self._make_transition_layer(input_features // 2, 2)

        input_features = input_features // 2
        self.densenetblock4 = self._make_dense_block(num_layers[3], input_features, 3)
        input_features += num_layers[3] * self.growth_rate

        # 输出部分
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(input_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_dense_block(self, num_layers, input_features, block_idx):
        layers = []
        for i in range(num_layers):
            layers.append(DenseNetBlock(input_features + i * self.growth_rate, self.growth_rate))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, input_features, block_idx):
        return TransitionLayer(input_features)

    def forward(self, x):
        out = self.Inputs(x)

        out = self.densenetblock1(out)
        out = self.transition1(out)

        out = self.densenetblock2(out)
        out = self.transition2(out)

        out = self.densenetblock3(out)
        out = self.transition3(out)

        out = self.densenetblock4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        out = self.softmax(out)

        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):

    def __init__(self, args, num_classes=10):
        super(MobileNet, self).__init__()
        self.cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out