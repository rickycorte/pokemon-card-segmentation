import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics as tm
import torch.nn.functional as F

class SimoidSegmentationModule(pl.LightningModule):
    """
    Common lightning that supporst binary segmentation models.
    
    Model should return raw predictions (no final activation function) with a depth of 1.
    
    Sigmoid function will be applyed internally
    """
    def __init__(self, model: nn.Module, lr: float=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr


        self.iou = tm.JaccardIndex(task="binary")
        self.valid_iou = tm.JaccardIndex(task="binary")

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = F.binary_cross_entropy(torch.sigmoid(pred), y)
        #loss = tops.sigmoid_focal_loss(pred, y, gamma=4, reduction="mean")

        self.iou(pred, y)
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        self.log("iou", self.iou, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = F.binary_cross_entropy(torch.sigmoid(pred), y)
        #loss = tops.sigmoid_focal_loss(pred, y, reduction="mean")

        self.valid_iou(pred, y)
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True)
        self.log("valid_iou", self.valid_iou, prog_bar=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        self.iou.reset()

    def on_validation_epoch_end(self):
        self.valid_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

