import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class MobileNet(nn.Module):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(4096, num_classes)
    
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layers(x)
        #print(x.shape)
        x = F.avg_pool2d(x, 7)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.linear(x)
        return x

class MobileNet_Classification(nn.Module):
    def __init__(self, class_num=102):
        super(MobileNet_Classification, self).__init__()
        self.features = self.get_mobilenet()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def get_mobilenet(self, class_num=102):
        model = MobileNet()
        channels_in = model.linear.in_features
        model.linear = nn.Linear(channels_in, class_num)
        return model
    
    def forward(self, x):
        x = self.features(x)
        x = self.log_softmax(x)
        return x


import torchvision
import numpy as np 
import os, random
from xml.dom.minidom import Document, parse
from PIL import Image
from torchvision.transforms import functional as Fun
import time

def readXml(filepath):
    domTree = parse(filepath)
    rootNode = domTree.documentElement
    object_node = rootNode.getElementsByTagName("class")[0]
    object_cls = object_node.childNodes[0].data
    return object_cls

def nll_loss(log_softmax, labels):
    loss_fn = torch.nn.NLLLoss(reduction="mean")
    loss = loss_fn(log_softmax, labels)
    #print(loss)
    return loss

def train_mbgd():
    cls_list = os.listdir("./101_ObjectCategories")
    Net = MobileNet_Classification(class_num=102)
    print(Net)
    #in_channels = Net.fc.in_features
    #Net.fc = nn.Linear(in_channels, 102)
    #Net = ResNet_Offical()
    torch.cuda.set_device(2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    Net.to(device)
    img_list = sorted(os.listdir("./dataset/images"))
    anno_list = sorted(os.listdir("./dataset/annotations"))

    optimizer = torch.optim.SGD(Net.parameters(), lr=0.005,momentum=0.9, weight_decay=0.0005)
    batch_size = 20
    num_epochs = 10

    for epoch in range(num_epochs):
        for idx in range(0,7000,batch_size):
            img_path  = [os.path.join("./dataset/images", img_list[i]) for i in range(idx,idx+batch_size)]
            anno_path = [os.path.join("./dataset/annotations", anno_list[i]) for i in range(idx,idx+batch_size)]
            img  = [Image.open(path).convert("RGB") for path in img_path]
            anno = [cls_list.index(readXml(path)) for path in anno_path]
            labels = torch.autograd.Variable(torch.Tensor(anno)).to(device).long()
            #func = ToTensor()
            #img, labels = func(img, labels)
            #img = img.to(device)
            img_var = [Fun.to_tensor(image) for image in img]
            img = torch.stack(img_var, dim=0).to(device)
            x = Net.forward(img)
            loss = nll_loss(x, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[%d]loss: %f" %(idx, loss))
        print("epoch[%d] finished." %epoch)

    torch.save(Net.state_dict(), "./mobileNetV1_model.pth")

def evaluate():
    torch.cuda.set_device(2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    Net = MobileNet_Classification(class_num=102)
    #Net = ResNet_Offical()
    Net.load_state_dict(torch.load("./mobileNetV1_model.pth"))
    Net.eval()
    Net.to(device)

    cls_list = os.listdir("./101_ObjectCategories")
    img_list = sorted(os.listdir("./dataset/images"))
    anno_list = sorted(os.listdir("./dataset/annotations"))

    num = 0
    acc_num = 0

    start_time = time.time()

    for idx in range(7000, len(img_list)):
        img_path = os.path.join("./dataset/images", img_list[idx])
        anno_path = os.path.join("./dataset/annotations", anno_list[idx])
        labels = cls_list.index(readXml(anno_path))
        img = Image.open(img_path).convert("RGB")
        img = Fun.to_tensor(img)
        img_var = torch.unsqueeze(img, dim=0).to(device)

        x = Net.forward(img_var)
        pred_cls = torch.argmax(x).item()
        print(idx,cls_list[pred_cls])
        if labels == pred_cls:
            acc_num += 1
        num += 1
    
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost time: %f" %cost_time)
    print("pre image cost time: %f" %(cost_time/num))
    print(acc_num)
    print(num)
    print(acc_num/num)
'''
from thop import profile, clever_format

if __name__ == '__main__':
    #train_mbgd()
    #evaluate()
    #model = MobileNet()
    model = MobileNet_Classification()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
'''

'''
flops: 2.312G
params: 3.625M
'''
'''
cost time: 14.420410
pre image cost time: 0.006723
1302
2145
accuracy：0.606993006993007
'''