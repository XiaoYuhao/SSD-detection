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
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}



def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.
    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    if not os.path.exists(output_folder):
    	os.makedirs(output_folder, exist_ok=True)
    # Training data
    for path in [voc07_path, voc12_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # Validation data
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in validation data
    with open(os.path.join(voc07_path, 'ImageSets/Main/val.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))

class PascalVOCDataset(Dataset):
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
    
if __name__ == '__main__':
    voc07_path = '../../dataset/voc/VOCdevkit/VOC2007'
    voc12_path = '../../dataset/voc/VOCdevkit/VOC2012'
    output_folder = '../../dataset/voc/'
    create_data_lists(voc07_path, voc12_path, output_folder)