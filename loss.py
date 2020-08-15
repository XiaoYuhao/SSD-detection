import torch
import torch.nn as nn
import torch.nn.functional as functional
from utils import *

class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratios=3, alpha=1., device='cuda'):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratios = neg_pos_ratios
        self.alpha = alpha
        self.device = device

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.float).to(device)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            overlap = find_jaccard_overlap(boxes[i],self.priors_xy)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)      #与每个先验框IoU值最大的GT
            _, prior_for_each_object = overlap.max(dim=1)                           #与每个GT框IoU值最大的先验框

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)      #最大IoU值先验框匹配相应GT框(0~n_object)  
            overlap_for_each_prior[prior_for_each_object] = 1                                                       #将具有最大IoU值的先验框IoU值置为1

            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0                                       #IoU小于阈值的类别置为背景

            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
        
        positive_priors = true_classes != 0     #取正样本
         
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = self.neg_pos_ratios * n_positives

        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1,).long())
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)

        conf_loss_pos = conf_loss_all[positive_priors]

        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()

        return conf_loss + self.alpha * loc_loss



