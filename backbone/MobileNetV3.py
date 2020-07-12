import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True)/6
        return out
    
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x+3, inplace=True)/6
        return out

class relu(nn.Module):
    def forward(self, x):
        out = F.relu(x)
        return out

class SeModule(nn.Module):
    def __init__(self, in_planes, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_planes//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//reduction, in_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_planes),
            hsigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    def __init__(self, kernel_size, in_planes, expand_planes, out_planes, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_planes, expand_planes, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(expand_planes)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_planes, expand_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.short_cut = nn.Sequential()
        #若stride>1说明输出尺寸会变小（下采样），若输入通道数和输出通道数不一致，则需要使用1X1卷积改变维度
        if stride == 1 and in_planes != out_planes:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nolinear1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nolinear2(out)
        if self.se != None:
            out = self.se(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.short_cut(x) if self.stride == 1 else out
        return out 

class MobileNetV3_Large(nn.Module):
    def __init__(self, class_num=102):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, relu(), None, 1),
            Block(3, 16, 64, 24, relu(), None, 2),
            Block(3, 24, 72, 24, relu(), None, 1),
            Block(5, 24, 72, 40, relu(), SeModule(72), 2),
            Block(5, 40, 120, 40, relu(), SeModule(120), 1),
            Block(5, 40, 120, 40, relu(), SeModule(120), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(480), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(672), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(672), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),
        )
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.hs3 = hswish()
        self.conv4 = nn.Conv2d(1280, class_num, kernel_size=1, stride=1, padding=0, bias=False)
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.hs1(out)
        out = self.bneck(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.hs2(out)
        out = self.avg_pool(out)
        #out = out.view(out.size(0), -1)
        out = self.conv3(out)
        out = self.hs3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        return out

class MobileNetV3_Small(nn.Module):
    def __init__(self, class_num=102):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, relu(), SeModule(16), 2),
            Block(3, 16, 72, 24, relu(), None, 2),
            Block(3, 24, 88, 24, relu(), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(96), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(240), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(240), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(120), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(144), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(288), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(576), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(576), 1),
        )
        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(576, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.hs3 = hswish()
        self.conv4 = nn.Conv2d(1024, class_num, kernel_size=1, stride=1, padding=0, bias=False)
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.hs1(out)
        out = self.bneck(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.hs2(out)
        out = self.avg_pool(out)
        #out = out.view(out.size(0), -1)
        out = self.conv3(out)
        out = self.hs3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        return out


class MobileNetV3_Large_Classification(nn.Module):
    def __init__(self, class_num=102):
        super(MobileNetV3_Large_Classification, self).__init__()
        self.features = MobileNetV3_Large(class_num)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.log_softmax(x)
        return x

class MobileNetV3_Small_Classification(nn.Module):
    def __init__(self, class_num=102):
        super(MobileNetV3_Small_Classification, self).__init__()
        self.features = MobileNetV3_Small(class_num)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.log_softmax(x)
        return x

from thop import profile, clever_format

if __name__ == '__main__':
    model = MobileNetV3_Small()
    print(model)
    image = torch.randn(2, 3, 300, 300)
    model(image)
    '''
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    '''


'''
MobileNetV3_Large_Classification:
flops: 232.705M
params: 4.349M

MobileNetV3_Small_Classification:
flops: 63.570M
params: 1.623M
'''