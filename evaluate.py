from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
from priors import *
import torch.nn.functional as F
from SSD import SSD
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#from voc0712 import PascalVOCDataset
from dataset import SeaShipsDataset
from tqdm import tqdm

def evaluate():
    backbone = 'VGG'
    model = SSD(class_num=7, backbone=backbone, device=device)
    model = load_pretrained(model, 'weights/ssd300_params_vgg_seaship_best.pth')

    #checkpoint = torch.load('checkpoint_ssd300.pth.tar')
    #model = checkpoint['model']
    model = model.to(device)
    model.eval()
    
    data_folder = 'dataset/SeaShips'
    keep_difficult = True
    batch_size = 2
    workers = 4
    test_dataset = SeaShipsDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

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
            bbox_results.extend([bbox2result(det_bboxes, det_labels, 7) for det_bboxes, det_labels in results])
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
    

