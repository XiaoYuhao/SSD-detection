import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        #1X1卷积用于提升通道数
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #分组卷积，组数为通道数，大大减少参数量和计算量
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        #如果输入通道和输出通道不一致，则shortcut需要经过1X1卷积提升维度
        if stride == 1 and in_planes != out_planes: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu6(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

class MobileNetV2(nn.Module):
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 2),
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
    
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu6(x)
        x = self.layers(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu6(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class MobileNetV2_Classification(nn.Module):
    def __init__(self, class_num=102):
        super(MobileNetV2_Classification, self).__init__()
        self.features = MobileNetV2(class_num)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.log_softmax(x)
        return x
        

from thop import profile, clever_format

if __name__ == '__main__':
    model = MobileNetV2_Classification()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

'''
flops: 337.235M
params: 2.414M
'''

