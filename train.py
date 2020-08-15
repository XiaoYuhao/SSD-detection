import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from loss import MultiBoxLoss
from utils import *
from SSD import SSD
from SSD512 import SSD512
from SSDLite import SSDLite
import argparse
from logger import getLogger
print(configs)

from dataset import Dataset

logger = getLogger(log_name=configs['log_name'])            #加载日志器

def train(train_loader, model, criterion, optimizer, epoch, grad_clip):
    print_freq = 200
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i , (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)
        
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip is not None:                   #用于防止梯度爆炸
            clip_gradient(optimizer, grad_clip)

        optimizer.step()
        
        #print(loss.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()
        #print('%d done...' %i)
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            logger.debug('Epoch: [{0}][{1}/{2}]\t'
                         'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                         'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            

def validate(val_loader, model, criterion):
    print_freq = 200
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(val_loader):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs, predicted_scores = model(images)
            
            predicted_locs = predicted_locs.to(device)
            predicted_scores = predicted_scores.to(device)

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
                logger.debug('[{0}/{1}]\t'
                             'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
            
            logger.debug('\n * Loss - {loss.avg:.3f}\n'.format(loss=losses))

            return losses.avg

keep_difficult = True  # use objects considered difficult to detect?

def main():
    n_classes = len(label_map)
    logger.debug(n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = configs['checkpoint']
    batch_size = configs['batch_size']
    start_epoch = configs['start_epoch']               #开始的epoch
    epochs = configs['epochs']                           #本次训练的epoch
    epochs_since_improvement = configs['epochs_since_improvement'] 
    best_loss = configs['best_loss']
    num_workers = configs['num_workers']
    lr = configs['lr']
    momentum = configs['momentum']
    weight_decay = configs['weight_decay']
    grad_clip = configs['grad_clip']
    backbone = configs['backbone']
    best_save = configs['best_model']
    save_model = configs['save_model']
    
    if configs['net'] == 'SSD':
        model = SSD(class_num=n_classes, backbone=backbone, device=device)
    elif configs['net'] == 'SSD512':
        model = SSD512(class_num=n_classes, backbone=backbone, device=device)
    elif configs['net'] == 'SSDLite':
        model = SSDLite(class_num=n_classes, backbone=backbone, device=device)
    if checkpoint is not None:
        model = load_pretrained(model, checkpoint, device=device)        #加载预训练模型

    data_folder = configs['data_folder']

    val_dataset = Dataset(data_folder, split='test', keep_difficult=keep_difficult)
    train_dataset = Dataset(data_folder, split='train', keep_difficult=keep_difficult)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=val_dataset.collate_fn, num_workers=num_workers, pin_memory=True)

    biases = list()
    not_biases = list()
    param_names_biases = list()
    param_names_not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
                param_names_biases.append(param_name)
            else:
                not_biases.append(param)
                param_names_not_biases.append(param_name)
    #optimizer = torch.optim.SGD(params=[{'params':biases,'lr': 2*lr}, {'params':not_biases}],
    #                            lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors).to(device)

    print(start_epoch)
    logger.debug(start_epoch)
    logger.debug(backbone)
    
    for epoch in range(start_epoch, epochs):
        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, grad_clip=grad_clip)

        val_loss = validate(val_loader=val_loader, model=model, criterion=criterion)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print('\nEpochs since last improvment: %d\n' %(epochs_since_improvement))
            logger.debug('\nEpochs since last improvment: %d\n' %(epochs_since_improvement))
        else:
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_save)
        
    torch.save(model.state_dict(), save_model)    
    logger.debug("End of training.")


if __name__ == '__main__':
    main()
    #save_model()








            
            

