from pycocotools.coco import COCO
import os
import os.path
import sys
import torch
from torch.utils.data import Dataset 
#import torchvision.transforms as transforms
from utils import transform
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
'''
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',*
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')
'''
COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


COCO_CLASSES ={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


class COCODataset(Dataset):
    def __init__(self, annotation_file, image_folder, split):
        self.annotation_file = annotation_file
        self.image_folder = image_folder
        self.split = split.upper()
        self.coco = COCO(self.annotation_file)
        
        img_ids = set(self.coco.imgToAnns.keys())
        for k, v in self.coco.anns.items():
            if(v['area'] < 20.0):
                if(v['image_id'] in img_ids):
                    img_ids.remove(v['image_id'])
        self.ids = list(img_ids)
        
        #self.ids = list(self.coco.imgToAnns.keys())
        self.len = len(self.ids)
        pass

    def __getitem__(self, i):
        _id = self.ids[i]
        image_path = os.path.join(self.image_folder ,  self.coco.imgs[_id]["file_name"])
        image = Image.open(image_path, mode='r')
        image = image.convert('RGB')
        
        ann_ids = self.coco.getAnnIds(imgIds=_id)
        target = self.coco.loadAnns(ann_ids)

        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]

        boxes = []
        labels = []

        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                label = COCO_LABEL_MAP[obj['category_id']]
                labels.append(label)

        
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        difficulties = torch.zeros_like(labels)
        
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        return image, boxes, labels, difficulties


    def __len__(self):
        return self.len

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
        
        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties

'''
coco = COCO(annotation_file='../../dataset/coco/annotations/instances_train2017.json')
all_ids = coco.imgs.keys()
print(len(all_ids))
all_ann = coco.anns.keys()
print(len(all_ann))
#print(all_ids)
'''

def darw_bbox_label(image, target):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Arial.ttf', 15)
    for obj in target:
        if 'bbox' in obj:
            bbox = obj['bbox']
            #boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            #label = COCO_LABEL_MAP[obj['category_id']] - 1
            #labels.append(label)

            draw.rectangle(xy=[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

            text_size = font.getsize(COCO_CLASSES[obj['category_id']].upper())
            text_location = [bbox[0] + 2., bbox[1] - text_size[1]]
            textbox_location = [bbox[0], bbox[1] - text_size[1], bbox[0] + text_size[0] + 4., bbox[1]]
            draw.rectangle(xy=textbox_location, fill="blue")
            draw.text(xy=text_location, text=COCO_CLASSES[obj['category_id']].upper(), fill='white', font=font)

        else:
            print("No bbox found for object ", obj)
    
    del draw

    image.save('temp/res002.jpg')

'''
dataset = COCODataset("../../dataset/coco/annotations/instances_train2017.json", "../../dataset/coco/train2017", split="TRAIN")

for i in range(dataset.__len__()):
    dataset.__getitem__(i)
'''

if __name__ == '__main__':
    pass