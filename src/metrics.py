# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import math
import torch
from sklearn.metrics import accuracy_score, f1_score

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.accuracy = 0
        self.PSNR = 0
        self.RMS = 0

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    #saeed
    def calc_accuracy(self, objects_true, objects_predicted):
        objects_predicted_indx_1 = objects_predicted>= 0.5
        objects_predicted_indx_0 = objects_predicted<0.5
        objects_predicted[objects_predicted_indx_1] = 1
        objects_predicted[objects_predicted_indx_0] = 0
        acc = f1_score(objects_true.squeeze(1)[0], objects_predicted[0],average='macro')
        return acc

    def calc_psnr(self, img1, img2):
        img1_size = img1.size()
        img1 = torch.clamp(img1, 0.0, 1.0)
        #img2 = torch.clamp(img2, 0.0, 1.0)
        # Debugging
        mese = (torch.mean((img1 - img2) ** 2))
        psnr_here = -10 * math.log10(mese.data)
        return psnr_here

    def calc_RMS(self, img1, img2):
        # calculate for the non-zero values of the original depth map
        #img2 orig
        #img1 pred
        #print (img2.shape)
        #print (img1.shape)
        mask = img2>0 #np.nonzero(img2)
        img2 = img2[mask]
        img1 = img1[mask]
        rms = np.sqrt((np.mean((img1 - img2) ** 2)))
        #psnr_here = -10 * math.log10(mese.data)
        #print mese.data, 
        #print psnr_here
        return rms
        

    def update(self, label_trues, label_preds ,image_orig , image_recon,depth_orig,depth_pred): #, objects_true, objects_predicted
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        self.PSNR += self.calc_psnr(image_recon, image_orig)
        self.RMS  += self.calc_RMS(depth_pred, depth_orig) 


    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - mean PSNR
            - mean RMS
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        self.PSNR     = float(self.PSNR) / (500)
        self.RMS     = float(self.RMS) / (500)

        return {'Mean IoU : \t': mean_iu,
                'PSNR : \t': self.PSNR, 
                'RMS : \t': self.RMS
                }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.PSNR = 0
        self.RMS = 0
