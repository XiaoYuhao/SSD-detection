from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
from priors import *
import torch.nn.functional as F
from utils import detect_objects
from SSD import SSD
from SSD512 import SSD512
from SSDLite import SSDLite
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = len(label_map)

def detect(img_path):
    print(n_classes)
    original_image = Image.open(img_path, mode='r').convert('RGB')
    backbone = test_configs['backbone']
    if test_configs['net'] == 'SSD':
        model = SSD(class_num=n_classes, backbone=backbone, device=device)
    elif test_configs['net'] == 'SSD512':
        model = SSD512(class_num=n_classes, backbone=backbone, device=device)
    elif test_configs['net'] == 'SSDLite':
        model = SSDLite(class_num=n_classes, backbone=backbone, device=device)
    model = load_pretrained(model, test_configs['checkpoint'], device=device)
    model.to(device)
    model.eval()

    resize = transforms.Resize(test_configs['resize'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)

    predicted_locs, predicted_scores = model(image.unsqueeze(0))


    det_boxes, det_labels, det_scores = detect_objects(model.priors, predicted_locs, predicted_scores, 0.10, 0.45, 200, n_classes)
    print(len(det_labels[0]))

    det_boxes = det_boxes[0].to('cpu')

    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels == ['background']:
        print("only background...")
        return original_image
    
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype('Arial.ttf', 15)

    for i in range(det_boxes.size(0)):
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])

        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)

    del draw
    
    annotated_image.save('temp/res001.jpg')
    return annotated_image

if __name__ == '__main__':
    detect('temp/test001.jpg')