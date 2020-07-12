from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
from priors import *
import torch.nn.functional as F
from SSD import SSD
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#priors_cxcy = generate_ssd_priors(mobilenet_specs, image_size)
#priors_cxcy = priors_cxcy.to(device)
n_classes = 20

#from voc0712 import PascalVOCDataset
from dataset import SeaShipsDataset
from tqdm import tqdm
from pprint import PrettyPrinter

def evaluate():
    backbone = 'MobileNetV3_Large'
    model = SSD(class_num=7, backbone=backbone, device=device)
    model = load_pretrained(model, 'ssd300_params_v3_large_seaship_best.pth')

    #checkpoint = torch.load('checkpoint_ssd300.pth.tar')
    #model = checkpoint['model']
    model = model.to(device)
    model.eval()

    pp = PrettyPrinter()
    data_folder = 'dataset/SeaShips'
    keep_difficult = True
    batch_size = 2
    workers = 4
    test_dataset = SeaShipsDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

    det_bboxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_scores = list()
    true_difficulties = list()

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            #print(images.shape)
            images = images.to(device)
            predicted_locs, predicted_scores = model(images)
            
            #det_boxes_batch, det_labels_batch, det_scores_batch = detector(priors_cxcy=model.priors, predicted_locs=predicted_locs, predicted_scores=predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200, n_classes=len(label_map))
            results = detector(model.priors, predicted_locs, predicted_scores, 0.02, 0.45, 200)

            #det_bboxes, det_labels = results[0]
            det_bboxes_batch = []
            det_labels_batch = []
            det_scores_batch = []
            for bboxes, dlabels in results:
                det_bboxes_batch.append(bboxes[:,:-1])
                det_labels_batch.append(dlabels)
                det_scores_batch.append(bboxes[:,-1])
            #det_scores = det_bboxes[:,:,-1]
            #print(det_bboxes_batch[0])
            #print(det_labels_batch[0])
            #print(det_scores_batch[0])
            #print(det_scores.shape)

            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_bboxes.extend(det_bboxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        tt = time.time()
        APs, mAP = calculate_mAP(det_bboxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        now = time.time()
        print("map time:%f" %(now - tt))
        
    count = 0
    for dlabels in det_labels:
        count += dlabels.shape[0]
    print("object num: %d" %count)
    print(APs)
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)


def mAP():
    backbone = 'MobileNetV3_Small'
    model = SSD(class_num=7, backbone=backbone, device=device)
    model = load_pretrained(model, 'ssd300_params_v3_small_seaship_best.pth')

    #checkpoint = torch.load('checkpoint_ssd300.pth.tar')
    #model = checkpoint['model']
    model = model.to(device)
    model.eval()

    pp = PrettyPrinter()
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

        eval_map(bbox_results, annotations)
            


def test_iou():
    det_bboxes = list()
    gt_bboxes = list()
    for i in range(10000):
        det_bboxes.append([0.5161582,0.27177492,0.8170899,0.42869678,0.48385584])
        det_bboxes.append([0.5161582,0.27177492,0.8170899,0.42869678,0.48385584])
        gt_bboxes.append([0.51875,0.17870371,0.92864585,0.45462963])
    det_bboxes = np.array(det_bboxes)
    gt_bboxes = np.array(gt_bboxes)
    
    s_time = time.clock()
    iou = bboxes_iou(det_bboxes[:,:-1], gt_bboxes)
    print(time.clock() - s_time)
    
    s_time = time.clock()
    iou2 = bbox_overlaps(det_bboxes[:,:-1], gt_bboxes)
    print(time.clock() - s_time)
    
    


if __name__ == '__main__':
    #detect('temp/test001.jpg')
    evaluate()
    #mAP()
    #test_iou()
