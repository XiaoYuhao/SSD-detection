import torch
import torch.nn as nn
import torch.nn.functional as F
from priors import *
from backbone import *

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.init_conv2d()
    
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu6(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu6(x)
        return x

class liteConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(liteConv, self).__init__()
        hidden_dim = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.init_conv2d()
    
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu6(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu6(out)
        return out

    
class PredictionConvolutions(nn.Module):
    def __init__(self, class_num, backbone):
        super(PredictionConvolutions, self).__init__()
        self.class_num = class_num
        
        n_boxes = {
            'conv4_3': 4,
            'conv7': 6,
            'conv8_2': 6,
            'conv9_2': 6,
            'conv10_2': 4,
            'conv11_2': 4
        }

        if backbone == 'MobileNetV1':
            self.loc_conv4_3 = Block(512, n_boxes['conv4_3']*4)
            self.loc_conv7 = Block(1024, n_boxes['conv7']*4)
            self.loc_conv8_2 = Block(512, n_boxes['conv8_2']*4)
            self.loc_conv9_2 = Block(256, n_boxes['conv9_2']*4)
            self.loc_conv10_2 = Block(256, n_boxes['conv10_2']*4)
            self.loc_conv11_2 = Block(128, n_boxes['conv11_2']*4)

            self.cl_conv4_3 = Block(512, n_boxes['conv4_3']*class_num)
            self.cl_conv7 = Block(1024, n_boxes['conv7']*class_num)
            self.cl_conv8_2 = Block(512, n_boxes['conv8_2']*class_num)
            self.cl_conv9_2 = Block(256, n_boxes['conv9_2']*class_num)
            self.cl_conv10_2 = Block(256, n_boxes['conv10_2']*class_num)
            self.cl_conv11_2 = Block(128, n_boxes['conv11_2']*class_num)
        
        elif backbone == 'MobileNetV2':
            self.loc_conv4_3 = Block(96, n_boxes['conv4_3']*4)
            self.loc_conv7 = Block(1280, n_boxes['conv7']*4)
            self.loc_conv8_2 = Block(512, n_boxes['conv8_2']*4)
            self.loc_conv9_2 = Block(256, n_boxes['conv9_2']*4)
            self.loc_conv10_2 = Block(256, n_boxes['conv10_2']*4)
            self.loc_conv11_2 = Block(256, n_boxes['conv11_2']*4)

            self.cl_conv4_3 = Block(96, n_boxes['conv4_3']*class_num)
            self.cl_conv7 = Block(1280, n_boxes['conv7']*class_num)
            self.cl_conv8_2 = Block(512, n_boxes['conv8_2']*class_num)
            self.cl_conv9_2 = Block(256, n_boxes['conv9_2']*class_num)
            self.cl_conv10_2 = Block(256, n_boxes['conv10_2']*class_num)
            self.cl_conv11_2 = Block(256, n_boxes['conv11_2']*class_num)

        
        elif backbone == 'MobileNetV3_Large':
            self.loc_conv4_3 = Block(112, n_boxes['conv4_3']*4)
            self.loc_conv7 = Block(960, n_boxes['conv7']*4)
            self.loc_conv8_2 = Block(512, n_boxes['conv8_2']*4)
            self.loc_conv9_2 = Block(256, n_boxes['conv9_2']*4)
            self.loc_conv10_2 = Block(256, n_boxes['conv10_2']*4)
            self.loc_conv11_2 = Block(256, n_boxes['conv11_2']*4)

            self.cl_conv4_3 = Block(112, n_boxes['conv4_3']*class_num)
            self.cl_conv7 = Block(960, n_boxes['conv7']*class_num)
            self.cl_conv8_2 = Block(512, n_boxes['conv8_2']*class_num)
            self.cl_conv9_2 = Block(256, n_boxes['conv9_2']*class_num)
            self.cl_conv10_2 = Block(256, n_boxes['conv10_2']*class_num)
            self.cl_conv11_2 = Block(256, n_boxes['conv11_2']*class_num)

        
        elif backbone == 'MobileNetV3_Small':
            self.loc_conv4_3 = Block(48, n_boxes['conv4_3']*4)
            self.loc_conv7 = Block(576, n_boxes['conv7']*4)
            self.loc_conv8_2 = Block(256, n_boxes['conv8_2']*4)
            self.loc_conv9_2 = Block(128, n_boxes['conv9_2']*4)
            self.loc_conv10_2 = Block(128, n_boxes['conv10_2']*4)
            self.loc_conv11_2 = Block(128, n_boxes['conv11_2']*4)

            self.cl_conv4_3 = Block(48, n_boxes['conv4_3']*class_num)
            self.cl_conv7 = Block(576, n_boxes['conv7']*class_num)
            self.cl_conv8_2 = Block(256, n_boxes['conv8_2']*class_num)
            self.cl_conv9_2 = Block(128, n_boxes['conv9_2']*class_num)
            self.cl_conv10_2 = Block(128, n_boxes['conv10_2']*class_num)
            self.cl_conv11_2 = Block(128, n_boxes['conv11_2']*class_num)

        elif backbone == 'VGG':
            self.loc_conv4_3 = Block(512, n_boxes['conv4_3']*4)
            self.loc_conv7 = Block(1024, n_boxes['conv7']*4)
            self.loc_conv8_2 = Block(512, n_boxes['conv8_2']*4)
            self.loc_conv9_2 = Block(256, n_boxes['conv9_2']*4)
            self.loc_conv10_2 = Block(256, n_boxes['conv10_2']*4)
            self.loc_conv11_2 = Block(128, n_boxes['conv11_2']*4)

            self.cl_conv4_3 = Block(512, n_boxes['conv4_3']*class_num)
            self.cl_conv7 = Block(1024, n_boxes['conv7']*class_num)
            self.cl_conv8_2 = Block(512, n_boxes['conv8_2']*class_num)
            self.cl_conv9_2 = Block(256, n_boxes['conv9_2']*class_num)
            self.cl_conv10_2 = Block(256, n_boxes['conv10_2']*class_num)
            self.cl_conv11_2 = Block(128, n_boxes['conv11_2']*class_num)
    
    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)

        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)

        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.class_num)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.class_num)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.class_num)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.class_num)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1 ,self.class_num)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.class_num)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        cls_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)
        #print(cls_scores.shape)

        return locs, cls_scores

class AuxillaryConvolutions(nn.Module):
    def __init__(self, backbone):
        super(AuxillaryConvolutions, self).__init__()
        self.backbone = backbone
        if backbone == 'MobileNetV1':
            self.extras = nn.ModuleList([
                liteConv(1024, 512, stride=2),
                liteConv(512, 256, stride=2),
                liteConv(256, 256, stride=2),
                liteConv(256, 128, stride=2),    
            ])

        elif backbone == 'MobileNetV2':
            self.extras = nn.ModuleList([
                liteConv(1280, 512, stride=2),
                liteConv(512, 256, stride=2),
                liteConv(256, 256, stride=2),
                liteConv(256, 256, stride=2),
            ])
        
        elif backbone == 'MobileNetV3_Large':
            self.extras = nn.ModuleList([
                liteConv(960, 512, stride=2),
                liteConv(512, 256, stride=2),
                liteConv(256, 256, stride=2),
                liteConv(256, 256, stride=2),
            ])

            self.init_conv2d()

        elif backbone == 'MobileNetV3_Small':
            self.extras = nn.ModuleList([
                liteConv(576, 256, stride=2),
                liteConv(256, 128, stride=2),
                liteConv(128, 128, stride=2),
                liteConv(128, 128, stride=2),
            ])

            self.init_conv2d()
        
        elif backbone == 'VGG':
            self.extras = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                    nn.ReLU()
                ),
                liteConv(1024, 512, stride=2),
                liteConv(512, 256, stride=2),
                liteConv(256, 256, stride=2),
                liteConv(256, 128, stride=3),
            ])

            self.init_conv2d()


    def init_conv2d(self):
        for c in self.children():
            for layer in c:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(c.bias, 0.)
    
    def forward(self, feats10x10):
        features = []
        x = feats10x10
        for layer in self.extras:
            x = layer(x)
            features.append(x)
        
        if self.backbone == 'VGG':
            features_19x19 = features[0]
            features_10x10 = features[1]
            features_5x5 = features[2]
            features_3x3 = features[3]
            features_1x1 = features[4]
            return features_19x19, features_10x10, features_5x5, features_3x3, features_1x1
        else:
            features_5x5 = features[0]
            features_3x3 = features[1]
            features_2x2 = features[2]
            features_1x1 = features[3]
            return features_5x5, features_3x3, features_2x2, features_1x1


class SSDLite(nn.Module):
    def __init__(self, class_num, backbone, device):
        super(SSDLite, self).__init__()
        self.class_num = class_num
        self.backbone = backbone
        if self.backbone == 'VGG':
            priors = generate_ssd_priors(vgg_specs, image_size)
        else:
            priors = generate_ssd_priors(mobilenet_specs, image_size)

        self.priors = torch.FloatTensor(priors).to(device)

        if self.backbone == 'MobileNetV1':
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.base_net = MobileNet()
        elif self.backbone == 'MobileNetV2':
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.last_conv =nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
            self.last_bn = nn.BatchNorm2d(1280) 
            self.base_net = MobileNetV2()
        elif self.backbone == 'MobileNetV3_Large':
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.hs1 = hswish()
            self.last_conv = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
            self.last_bn = nn.BatchNorm2d(960)
            self.last_hs = hswish()
            self.base_net = MobileNetV3_Large()
        elif self.backbone == 'MobileNetV3_Small':
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.hs1 = hswish()
            self.last_conv = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
            self.last_bn = nn.BatchNorm2d(576)
            self.last_hs = hswish()
            self.base_net = MobileNetV3_Small()
        elif self.backbone == 'VGG':
            self.base_net = VGG()


        
        
        self.aux_net = AuxillaryConvolutions(backbone=self.backbone)
        self.prediction_net = PredictionConvolutions(class_num=self.class_num, backbone=self.backbone)


    
    def forward(self, x):
        if self.backbone == 'MobileNetV1':
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            for index, feat in enumerate(self.base_net.layers):
                x = feat(x)
                if index == 10:
                    features_19x19 = x
                    #print(features_19x19.shape)
                if index == 12:
                    features_10x10 = x
                    #print(features_10x10.shape)
        elif self.backbone == 'MobileNetV2':
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            for index, feat in enumerate(self.base_net.layers):
                x = feat(x)
                #print(x.shape)
                if index == 11:
                    features_19x19 = x
                if index == 16:
                    x = self.last_conv(x)
                    x = self.last_bn(x)
                    x = F.relu(x)
                    features_10x10 = x

        elif self.backbone == 'MobileNetV3_Large':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.hs1(x)
            for index, feat in enumerate(self.base_net.bneck):
                x = feat(x)
                if index == 10:
                    features_19x19 = x
                if index == 14:
                    x = self.last_conv(x)
                    x = self.last_bn(x)
                    x = self.last_hs(x)
                    features_10x10 = x

        elif self.backbone == 'MobileNetV3_Small':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.hs1(x)
            for index, feat in enumerate(self.base_net.bneck):
                x = feat(x)
                if index == 7:
                    features_19x19 = x
                if index == 10:
                    x = self.last_conv(x)
                    x = self.last_bn(x)
                    x = self.last_hs(x)
                    features_10x10 = x
        
        elif self.backbone == 'VGG':
            for index, feat in enumerate(self.base_net.features):
                x = feat(x)
                if index == 32:
                    features_19x19 = x      #feature size is actually 38*38
                if index == 42:
                    break
                 


        if self.backbone == 'VGG':
            #feature size is actually 19x19 10x10 5x5 3x3 1x1
            #print("VGG")
            features_10x10, features_5x5, features_3x3, features_2x2, features_1x1 = self.aux_net(x)
        else:
            features_5x5, features_3x3, features_2x2, features_1x1 = self.aux_net(x)

        #print(features_19x19.shape)
        #print(features_10x10.shape)
        #print(features_5x5.shape)
        #print(features_3x3.shape)
        #print(features_2x2.shape)
        #print(features_1x1.shape)

        features = []
        features.append(features_19x19)
        features.append(features_10x10)
        features.append(features_5x5)
        features.append(features_3x3)
        features.append(features_2x2)
        features.append(features_1x1)

        locs, cls_scores = self.prediction_net(features_19x19, features_10x10, features_5x5, features_3x3, features_2x2, features_1x1)
        #print(locs.shape)
        #print(cls_scores.shape)

        return locs, cls_scores

#from thop import profile, clever_format
from torchstat import stat

if __name__ == '__main__':
    
    #M = MobileNet()
    #for index, feat in enumerate(M.layers):
    #    print(index)
    #    print(feat)
    '''
    model = SSD(class_num=7, backbone='VGG', device='cpu')
    print(model)
    image = torch.randn(3, 3, 300, 300)
    model(image)
    '''
    #torch.save(model.state_dict(), 'temp.pth')
    '''
    image = torch.randn(1, 3, 300, 300)
    model = SSD(class_num=6, backbone='MobileNetV1', device='cpu')
    flops, params = profile(model, inputs=(image,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    '''

    
    model = SSDLite(class_num=7, backbone='MobileNetV1', device='cpu')
    stat(model, (3, 300, 300))
    
    
