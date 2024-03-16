import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CyclicLR

import lightning as L
from torchmetrics.classification import MultilabelAccuracy
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from base_dataPreprocess import MotionDataset, MotionDataModule

parser = argparse.ArgumentParser()
parser.add_argument('--isTrain',
                    action='store_true',
                    help='does it start to training')
parser.add_argument(
    "--annotations_file",
    type=str,
    default='',
    help=
    "the annotations file path(with openpose csv path, angle csv path, and label)"
)
parser.add_argument("--n_epochs",
                    type=int,
                    default=100,
                    help="number of epochs of training")
parser.add_argument("--batch_size",
                    type=int,
                    default=16,
                    help="size of the batches")
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="adam: learning rate")
parser.add_argument("--base_lr",
                    type=int,
                    default=1e-5,
                    help="the base learning rate of the cyclic learning rate")
parser.add_argument("--max_lr",
                    type=float,
                    default=1e-3,
                    help="the max learning rate of the cyclic learning rate")
parser.add_argument("--b1",
                    type=float,
                    default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",
                    type=float,
                    default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=4,
    help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval",
                    type=int,
                    default=400,
                    help="interval betwen image samples")
parser.add_argument("--num_labels",
                    type=int,
                    default=6,
                    help="the number of label")
parser.add_argument("--input_dim",
                    type=int,
                    default=34,
                    help="the input dimension of the lstm")
parser.add_argument("--hidden_dim",
                    type=int,
                    default=256,
                    help="the hidden dimension of the lstm")
parser.add_argument("--layer_dim",
                    type=int,
                    default=3,
                    help="the layer dimension of the lstm")
parser.add_argument(
    "--label_threshold",
    type=float,
    default=0.5,
    help=
    "Threshold for transforming probability to binary (0,1) predictions in label"
)
parser.add_argument("--prev_ckpt_path",
                    type=str,
                    default=None,
                    help="load previous checkpoint for further training")
opt = parser.parse_args()


class LSTMClassifier(L.LightningModule):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dm):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim,
                           hidden_dim,
                           layer_dim,
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(
            hidden_dim * 2,
            output_dim)  # Multiply hidden_dim by 2 due to bidirectional LSTM
        self.sigmoid = nn.Sigmoid()

        self.loss = None
        self.accuracy = MultilabelAccuracy(num_labels=opt.num_labels,
                                           threshold=opt.label_threshold)
        self.dm = dm
        # Save hyperparameters
        self.save_hyperparameters({
            'n_epochs': opt.n_epochs,
            'batch_size': opt.batch_size,
            'lr': opt.lr,
            'base_lr': opt.base_lr,
            'max_lr': opt.max_lr,
            'n_cpu': opt.n_cpu,
            'num_labels': opt.num_labels,
            'input_dim': opt.input_dim,
            'hidden_dim': opt.hidden_dim,
            'layer_dim': opt.layer_dim,
            'label_threshold': opt.label_threshold,
            'output_dim': output_dim,
        })

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Extract the last time step output
        out = self.sigmoid(out)
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

    def training_step(self, batch, batch_idx):
        # Training step
        self.train()
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.loss = loss
        self.log('train_loss_step',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.accuracy(y_hat, y)
        self.log('train_acc_step',
                 self.accuracy,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: int,
                           batch_idx: int) -> None:
        self.log('train_loss_epoch',
                 self.loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_acc_epoch',
                 self.accuracy,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        if batch_idx % 100 == 0:
            self.eval()
            with torch.no_grad():
                test_dataloader = self.dm.test_dataloader()
                for test_batch in test_dataloader:
                    x, y = test_batch
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self(x)
                    loss = F.binary_cross_entropy(y_hat, y)
                    self.log('test_loss_epoch',
                             loss,
                             on_step=True,
                             on_epoch=True,
                             prog_bar=True,
                             logger=True)
                    self.accuracy(y_hat, y)
                    self.log('test_acc_epoch',
                             self.accuracy,
                             on_step=True,
                             on_epoch=True,
                             prog_bar=True,
                             logger=True)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('final_test_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        pred = self(batch)
        return pred

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr)
        lr_scheduler_config = {
            "scheduler":
            CyclicLR(optimizer,
                     base_lr=opt.base_lr,
                     max_lr=opt.max_lr,
                     cycle_momentum=False),
            "interval":
            "epoch",
            "frequency":
            1,
            "motitor":
            'val_loss'
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    @classmethod
    def lstm_forehand_predict(cls,
                              annotation_filepath,
                              prev_ckpt_path,
                              batch_size=1,
                              n_cpu=4):
        dm = MotionDataModule(annotations_file=annotation_filepath,
                              batch_size=batch_size,
                              n_cpu=n_cpu)
        model = LSTMClassifier.load_from_checkpoint(prev_ckpt_path,
                                                    output_dim=6,
                                                    dm=dm)
        trainer = L.Trainer(accelerator='auto')
        pred = trainer.predict(model, dm)
        return pred


if __name__ == "__main__":
    if opt.isTrain:
        ################### Train #####################
        dm = MotionDataModule(annotations_file=opt.annotations_file,
                              batch_size=opt.batch_size,
                              n_cpu=opt.n_cpu)
        model = LSTMClassifier(input_dim=opt.input_dim,
                               hidden_dim=opt.hidden_dim,
                               layer_dim=opt.layer_dim,
                               output_dim=opt.num_labels,
                               dm=dm)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=-1,
            monitor='train_loss_epoch',
            filename='resnet-{epoch:02d}-{train_loss:.4f}')
        logger = TensorBoardLogger("tb_logs")
        trainer = L.Trainer(accelerator='auto',
                            max_epochs=opt.n_epochs,
                            callbacks=[checkpoint_callback],
                            logger=logger)
        trainer.fit(model, dm)

        ################### Test #####################
        trainer.test(model, dm)

    else:
        ################### Predict #####################
        dm = MotionDataModule(annotations_file=opt.annotations_file,
                              batch_size=opt.batch_size,
                              n_cpu=opt.n_cpu)
        model = LSTMClassifier.load_from_checkpoint(opt.prev_ckpt_path)
        trainer = L.Trainer(accelerator='auto')
        pred = trainer.predict(model, dm)
        print(pred)
