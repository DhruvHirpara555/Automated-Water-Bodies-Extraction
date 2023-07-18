import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def precision(y_true, y_pred):
    TP = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    TP_FP = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    precision = TP / (TP_FP + 1e-7)
    return precision

def recall(y_true, y_pred):
    TP = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    TP_FN = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    recall = TP / (TP_FN + 1e-7)
    return recall


def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + 1e-7))

def adaptive_maxpool_loss(y_true, y_pred, alpha=0.25):
    y_pred = torch.clamp(y_pred, 1e-7, 1. - 1e-7)
    positive = -y_true * torch.log(y_pred) * alpha
    negative = -(1. - y_true) * torch.log(1. - y_pred) * (1 - alpha)
    pointwise_loss = positive + negative
    x = torch.mean(pointwise_loss)
    max_loss = F.max_pool2d(pointwise_loss, kernel_size=4, stride=1)
    max_loss = F.pad(max_loss, (1, 2, 1, 2))

    # max_loss = F.max_pool2d(pointwise_loss, kernel_size=8, stride=1, padding = [])
    x = torch.mul(pointwise_loss, max_loss)
    # x = torch.sum(x)/torch.sum(max_loss)
    x = torch.mean(x)
    return x