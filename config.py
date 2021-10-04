
#data_folder = 'dataset/SeaShips'
data_folder = 'coco2017'

configs = dict(
    net = 'SSD',
    log_name = 'ssd_mobilenetv2_coco',
    checkpoint = None,
    batch_size = 64,
    start_epoch = 0,
    epochs = 300,
    epochs_since_improvement = 0,
    best_loss = 100.,
    num_workers = 4,
    lr = 0.0005,
    momentum = 0.9,
    weight_decay = 0.0005,
    grad_clip = None,
    backbone = 'MobileNetV2',
    best_model = 'weights/ssd_mobilenetv2_coco_best.pth',
    save_model = 'weights/ssd_mobilenetv2_coco.pth',
    data_folder = data_folder,
    resize = (300, 300)
)

test_configs = dict( 
    net = 'SSD',
    checkpoint = 'weights/ssd_mobilenetv2_coco_best.pth',
    backbone = 'MobileNetV2',
    data_folder = data_folder,
    resize = (300, 300)
)

if data_folder == '../../dataset/voc':

    # Label map
    labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    label_map = {k: v + 1 for v, k in enumerate(labels)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    # Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                    '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                    '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

elif data_folder == 'dataset/SeaShips':
    
    labels = ['ore carrier','passenger ship','container ship','bulk cargo carrier','general cargo ship','fishing boat']
    label_map = {k: v + 1 for v, k in enumerate(labels)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                    '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                    '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

elif data_folder == 'coco2017':

    labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
    label_map = {k : v + 1 for v, k in enumerate(labels)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}


else:
    print("Data Error: data folder not found.")
    exit(0)