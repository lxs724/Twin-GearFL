import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import torch.nn.functional as F


class CIFAR10_model(nn.Module):
    def __init__(self):
        super(CIFAR10_model, self).__init__()
        # 卷积核属性：输入通道数：3(RGB) 输出通道：32，核：5*5          input: 3@32*32  output：32@32*32
        self.conv1 = Conv2d(3, 64, 3, padding=1)
        self.maxpool1 = MaxPool2d(2, padding=1)  # 最大池化层 核 2*2   input:32@32*32   output：32@16*16
        # self.conv2 = Conv2d(32, 32, 5, padding=2)  # input: 32@16*16  output：32@16*16       # out64
        self.conv2 = Conv2d(64, 128, 3, padding=1)
        self.maxpool2 = MaxPool2d(2, padding=1)  # 最大池化层 核 2*2   input:32@16*16   output：32@8*8
        # self.conv3 = Conv2d(32, 64, 5, padding=2)  # input: 32@8*8  output:64@8*8
        self.conv3 = Conv2d(128, 384, 3, padding=1)
        self.conv4 = Conv2d(384, 256, 3, padding=1)
        # self.maxpool3 = MaxPool2d(2)  # input: 64@8*8     output: 64@4*4
        self.conv5 = Conv2d(256, 256, 3, padding=1)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()  # input: 64@4*4     output:1024
        self.linear1 = Linear(4096, 64)  # input: 1024    output: 64
        self.linear2 = Linear(64, 10)  # input: 64     output: 10

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x = self.maxpool3(F.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.linear2(F.relu(self.linear1(x)))  # relu -> softmax
        return x



class CIFAR100_model(nn.Module):
    def __init__(self):
        super(CIFAR100_model, self).__init__()
        # 卷积核属性：输入通道数：3(RGB) 输出通道：32，核：5*5          input: 3@32*32  output：32@32*32
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)  # 最大池化层 核 2*2   input:32@32*32   output：32@16*16
        self.conv2 = Conv2d(32, 32, 5, padding=2)  # input: 32@16*16  output：32@16*16
        self.maxpool2 = MaxPool2d(2)  # 最大池化层 核 2*2   input:32@16*16   output：32@8*8
        self.conv3 = Conv2d(32, 64, 5, padding=2)  # input: 32@8*8  output:64@8*8
        self.maxpool3 = MaxPool2d(2)  # input: 64@8*8     output: 64@4*4
        self.flatten = Flatten()  # input: 64@4*4     output:1024
        self.linear1 = Linear(1024, 500)  # input: 1024    output: 64
        self.linear2 = Linear(500, 100)  # input: 64     output: 10

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.linear2(F.relu(self.linear1(x)))
        return x

class mnist_cnn(nn.Module):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNFashion_Mnist(nn.Module):
    def __init__(self):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SVHN_model(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(SVHN_model, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        self.block4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.outlayer = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)  # [200, 64, 28, 28]
        x = self.block2(x)  # [200, 128, 14, 14]
        x = self.block3(x)  # [200, 256, 7, 7]
        x = self.block4(x)
        x = F.avg_pool2d(x, 4)  # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [200,256]
        out = self.outlayer(x)
        return out

# class Basicblock(nn.Module):
#     def __init__(self, in_planes, planes, stride=1):
#         super(Basicblock, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(planes),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(planes),
#         )
#
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
#                 nn.BatchNorm2d(planes)
#             )
#         else:
#             self.shortcut = nn.Sequential()
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_block, num_classes):
#         super(ResNet, self).__init__()
#         self.in_planes = 16
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#
#         self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
#         self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
#         self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
#         # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)
#
#         self.outlayer = nn.Linear(64, num_classes)
#
#     def _make_layer(self, block, planes, num_block, stride):
#         layers = []
#         for i in range(num_block):
#             if i == 0:
#                 layers.append(block(self.in_planes, planes, stride))
#             else:
#                 layers.append(block(planes, planes, 1))
#         self.in_planes = planes
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.maxpool(self.conv1(x))
#         x = self.block1(x)  # [200, 64, 28, 28]
#         x = self.block2(x)  # [200, 128, 14, 14]
#         x = self.block3(x)  # [200, 256, 7, 7]
#         # out = self.block4(out)
#         x = F.avg_pool2d(x, 7)  # [200, 256, 1, 1]
#         x = x.view(x.size(0), -1)  # [200,256]
#         out = self.outlayer(x)
#         return out


if __name__ == "__main__":
    model = CIFAR10_model()
    print(model)
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)
