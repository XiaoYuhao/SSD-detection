from utils import *
from dataset import SeaShipsDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from SSD import SSD
from config import *
import time

pp = PrettyPrinter()
data_folder = 'dataset/SeaShips'
keep_difficult = True
batch_size = 2
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
model = SSD(class_num=len(label_map), backbone='MobileNetV2', device=device)
model = load_pretrained(model, 'ssd300_params_v2_best.pth')
#model.load_state_dict(torch.load('ssd300_params.pth'))
'''
backbone = 'MobileNetV3_Large'
model = SSD(class_num=7, backbone=backbone, device=device)
model = load_pretrained(model, 'ssd300_params_v3_large_seaship_best.pth')
model = model.to(device)
model.eval()

test_dataset = SeaShipsDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    model.eval()

    det_boxes = list()
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
            
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(priors_cxcy=model.priors, predicted_locs=predicted_locs, predicted_scores=predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200, n_classes=len(label_map))
            
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)

if __name__ == '__main__':
    evaluate(test_loader, model)
    





