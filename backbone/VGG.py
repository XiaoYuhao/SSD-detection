import torch
import torch.nn as nn
import torchvision
import numpy as np 

class VGG(nn.Module):
    def __init__(self, class_num=102):
        super(VGG, self).__init__()
        self.class_num = class_num
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._vgg_layers(cfg)
        #self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(7)
        self.classifier()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def _vgg_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x ,kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)
                        ]
                in_channels = x
            
        return nn.Sequential(*layers)
    
    def classifier(self, in_channels=512):
        self.cls_layer = nn.Sequential(
            *[nn.Linear(25088, 4096),
              nn.ReLU(),
              nn.Linear(4096,4096),
              nn.ReLU(),
              nn.Linear(4096,self.class_num)])
        

    def forward(self, data):
        #print(data.shape)
        out_map = self.features(data)
        #print(out_map.size())
        #x = self.adaptive_max_pool(out_map)
        #print(x.size())
        k = out_map.view(out_map.size(0), -1)
        #print(k.size())
        k = self.cls_layer(k)
        k = self.log_softmax(k)
        #print(k)
        return k


if __name__ == '__main__':
    model = VGG()
    print(model)

    offical_model = torchvision.models.vgg16()
    print(offical_model)
    

    


    