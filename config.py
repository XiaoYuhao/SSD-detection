
data_folder = 'dataset/SeaShips'

configs = dict(
    net = 'SSD512',
    log_name = 'ssd512_vgg_seaships',
    checkpoint = 'weights/ssd512_vgg_seaships.pth',
    batch_size = 10,
    start_epoch = 56,
    epochs = 100,
    epochs_since_improvement = 0,
    best_loss = 100.,
    num_workers = 4,
    lr = 0.005,
    momentum = 0.9,
    weight_decay = 0.0005,
    grad_clip = None,
    backbone = 'VGG',
    best_model = 'weights/ssd512_vgg_seaships_best.pth',
    save_model = 'weights/ssd512_vgg_seaships.pth',
    data_folder = data_folder,
    resize = (512, 512)
)

test_configs = dict( 
    net = 'SSD512',
    checkpoint = 'weights/ssd512_vgg_seaships_best.pth',
    backbone = 'VGG',
    data_folder = data_folder,
    resize = (512, 512)
)

if data_folder == 'dataset/voc':

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

else:
    print("Data Error: data folder not found.")
    exit(0)