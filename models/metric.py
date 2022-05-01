import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import chisquare

__all__ = ['ConfusionMatrix']

class ConfusionMatrix():

    def __init__(self, pred, target, is_prob=False):
        self.pom = []
        if is_prob:
            self.pom = F.softmax(pred, dim=1)[:,1].cpu()
            pred = pred.argmax(dim=1)
            pred[pred==2] = 0
        self.target = target.view_as(pred).cuda()
        
        self.tp = (pred * self.target).sum().item()
        self.fp = (pred * (1-self.target)).sum().item()
        self.fn = ((1-pred) * self.target).sum().item()
        self.tn = ((1-pred) * (1-self.target)).sum().item()

        self.pred = pred.cpu()
        
    def calc_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def calc_sensitivity(self):
        return self.tp / (self.tp + self.fn)

    def calc_specificity(self):
        return self.tn / (self.tn + self.fp)

    def calc_precision(self):
        return self.tp / (self.tp + self.fp)

    def calc_F1_score(self):
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def calc_roc_auc(self):
        if len(self.pom): return roc_auc_score(self.target.cpu(), self.pom)

    def calc_p_value(self):
        if len(self.pom): return chisquare(self.pom*10+5, self.target.cpu()*10+5).pvalue

    def __str__(self):
        return f'TP: {self.tp} TN: {self.tn} FP: {self.fp} FN: {self.fn}'
