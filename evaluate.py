from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
from priors import *
import torch.nn.functional as F
from SSD import SSD
from SSD512 import SSD512
from SSDLite import SSDLite
import time
from coco2017 import COCODataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

from dataset import Dataset
from tqdm import tqdm

def evaluate():
    print(device)
    backbone = test_configs['backbone']
    if test_configs['net'] == 'SSD':
        model = SSD(class_num=len(label_map), backbone=backbone, device=device)
    elif test_configs['net'] == 'SSD512':
        model = SSD512(class_num=len(label_map), backbone=backbone, device=device)
    elif test_configs['net'] == 'SSDLite':
        model = SSDLite(class_num=len(label_map), backbone=backbone, device=device)
    model = load_pretrained(model, test_configs['checkpoint'], device=device)

    print(len(label_map))
    #checkpoint = torch.load('checkpoint_ssd300.pth.tar')
    #model = checkpoint['model']
    model = model.to(device)
    model.eval()
    
    data_folder = test_configs['data_folder']
    keep_difficult = True
    batch_size = 2
    workers = 4
    pin_memory = False if device == torch.device('cpu') else True
    #test_dataset = Dataset(data_folder, split='test', keep_difficult=keep_difficult)
    test_dataset = COCODataset("../../dataset/coco/annotations/instances_val2017.json", "../../dataset/coco/val2017", split="TEST")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=pin_memory)

    annotations = list()
    bbox_results = list()

    with torch.no_grad():
        for i, (images, bboxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            #print(images.shape)
            images = images.to(device)
            predicted_locs, predicted_scores = model(images)
            
            #det_boxes_batch, det_labels_batch, det_scores_batch = detector(priors_cxcy=model.priors, predicted_locs=predicted_locs, predicted_scores=predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200, n_classes=len(label_map))
            results = detector(model.priors, predicted_locs, predicted_scores, 0.02, 0.45, 200)
            #print(results)
            bbox_results.extend([bbox2result(det_bboxes, det_labels, len(label_map)) for det_bboxes, det_labels in results])
            #print(bbox_results)

            #boxes = [b.to(device) for b in boxes]
            #labels = [l.to(device) for l in labels]

            for b,l in zip(bboxes, labels):
                anno = dict()
                anno['bboxes'] = b.cpu().detach().numpy()
                anno['labels'] = l.cpu().detach().numpy()
                annotations.append(anno)
            
            #print(annotations)

        mAP, eval_results = eval_map(bbox_results, annotations)
        show_mAP_table(eval_results, mAP)


if __name__ == '__main__':
    evaluate()
    

