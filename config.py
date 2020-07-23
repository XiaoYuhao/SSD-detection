
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


'''
labels = ['ore carrier','passenger ship','container ship','bulk cargo carrier','general cargo ship','fishing boat']
label_map = {k: v + 1 for v, k in enumerate(labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
'''

configs = dict(
    log_name = 'ssd300_v3_large_voc',
    checkpoint = None,
    batch_size = 20,
    start_epoch = 0,
    epochs = 100,
    epochs_since_improvement = 0,
    best_loss = 100.,
    num_workers = 4,
    lr = 0.002,
    momentum = 0.9,
    weight_decay = 0.0005,
    grad_clip = None,
    backbone = 'MobileNetV3_Large',
    best_model = 'ssd300_v3_large_voc_best.pth',
    save_model = 'ssd300_v3_large_voc.pth',
    data_folder = 'dataset/voc',
)

test_configs = dict(
    checkpoint = 'ssd300_v1_voc_best.pth',
    backbone = 'MobileNetV1',
    data_folder = 'dataset/voc'
)