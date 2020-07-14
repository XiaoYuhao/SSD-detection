import torch
from torch.utils.data import Dataset
import json, os
from PIL import Image
import xml.etree.ElementTree as ET
from utils import transform
from config import *

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for obj in root.iter('object'):

        difficult = int(obj.find('difficult').text == '1')

        label = obj.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def create_data_lists(data_path, output_folder):
    train_images = list()
    train_objects = list()
    n_objects = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    with open(os.path.join(data_path, 'ImageSets/Main/train.txt')) as f:
        ids = f.read().splitlines()
    
    for id in ids:
        objects = parse_annotation(os.path.join(data_path, 'Annotations', id + '.xml'))
        if len(objects['labels']) == 0:
            continue
        n_objects += len(objects['labels'])
        train_objects.append(objects)
        train_images.append(os.path.join(data_path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)
    
    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' %(len(train_images), n_objects, os.path.abspath(output_folder)))

    test_images = list()
    test_objects = list()
    n_objects = 0

    with open(os.path.join(data_path, 'ImageSets/Main/val.txt')) as f:
        ids = f.read().splitlines()
    
    for id in ids:
        objects = parse_annotation(os.path.join(data_path, 'Annotations', id + '.xml'))
        if len(objects['labels']) == 0:
            continue
        n_objects += len(objects['labels'])
        test_objects.append(objects)
        test_images.append(os.path.join(data_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)
    
    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' %(len(test_images), n_objects, os.path.abspath(output_folder)))

class Dataset(Dataset):
    def __init__(self, data_folder, split, keep_difficult=False):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        with open(os.path.join(data_folder, self.split+'_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split+'_objects.json'), 'r') as j:
            self.objects = json.load(j)
        
        assert len(self.images) == len(self.objects)
    
    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])
        labels = torch.LongTensor(objects['labels'])
        difficulties = torch.ByteTensor(objects['difficulties'])

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]
        
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
         
        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)
    
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


#data_path = '../coco/data/SeaShips/VOCdevkit/VOC2007'
#create_data_lists(data_path, output_folder=data_folder)
#data_folder = 'dataset'