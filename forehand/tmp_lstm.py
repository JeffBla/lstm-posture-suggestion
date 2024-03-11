import argparse
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR

import lightning as L
from torchmetrics.classification import MultilabelAccuracy
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint

from forehand.dataPreprocess import MotionDataset, MotionDataModule

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
                    default=32,
                    help="size of the batches")
parser.add_argument("--lr",
                    type=float,
                    default=0.02,
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
                    default=3,
                    help="the number of label")
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

# def accuracy(output, target):
#     return (output.argmax(dim=1) == target).float().mean().item()

# class CyclicLR(_LRScheduler):

#     def __init__(self, optimizer, schedule, last_epoch=-1):
#         assert callable(schedule)
#         self.schedule = schedule
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

# def cosine(t_max, eta_min=0):

#     def scheduler(epoch, base_lr):
#         t = epoch % t_max
#         return eta_min + (base_lr - eta_min) * (1 +
#                                                 np.cos(np.pi * t / t_max)) / 2

#     return scheduler

# n = 100
# sched = cosine(n)
# lrs = [sched(t, 1) for t in range(n * 4)]
# plt.plot(lrs)


class LSTMClassifier(L.LightningModule):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim,
                           hidden_dim,
                           layer_dim,
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.loss = None
        self.accuracy = MultilabelAccuracy(num_labels=opt.num_labels,
                                           threshold=opt.label_threshold)

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

    def training_step(self, batch, batch_idx):
        # Training step
        self.train()
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.loss = loss
        self.log('train_loss_step',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.accuracy(y_hat, y)
        self.log('train_acc_step', self.accuracy,on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: int,
                           batch_idx: int) -> None:
        self.log('train_loss_epoch', self.loss,on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_acc_epoch', self.accuracy,on_step=True,
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
                    loss = F.cross_entropy(y_hat, y)
                    self.log('test_loss_epoch',
                             loss,
                             on_step=True,
                             on_epoch=True,
                             prog_bar=True,
                             logger=True)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
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
            CyclicLR(optimizer, base_lr=opt.base_lr, max_lr=opt.max_lr),
            "interval":
            "epoch",
            "frequency":
            1,
            "motitor":
            'val_loss'
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


if opt.isTrain:
    ################### Train #####################
    dm = MotionDataModule(annotations_file=opt.annotations_file,
                          batch_size=opt.batch_size,
                          n_cpu=opt.n_cpu)
    model = LSTMClassifier(input_dim=10,
                           hidden_dim=256,
                           layer_dim=3,
                           output_dim=9)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor='train_loss',
        filename='resnet-{epoch:02d}-{train_loss:.4f}')
    trainer = L.Trainer(accelerator='auto',
                        max_epochs=opt.n_epochs,
                        callbacks=[checkpoint_callback])
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
    trainer.predict(model, dm)

input_dim = 10
hidden_dim = 256
layer_dim = 3
output_dim = 9
seq_dim = 128

lr = 0.0005
n_epochs = 1000
# iterations_per_epoch = len(trn_dl)
best_acc = 0
patience, trials = 100, 0

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
opt = torch.optim.RMSprop(model.parameters(), lr=lr)
# sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))

# print('Start model training')

# for epoch in range(1, n_epochs + 1):

#     for i, (x_batch, y_batch) in enumerate(trn_dl):
#         model.train()
#         x_batch = x_batch.cuda()
#         y_batch = y_batch.cuda()
#         sched.step()
#         opt.zero_grad()
#         out = model(x_batch)
#         loss = criterion(out, y_batch)
#         loss.backward()
#         opt.step()

#     model.eval()
#     correct, total = 0, 0
#     for x_val, y_val in val_dl:
#         x_val, y_val = [t.cuda() for t in (x_val, y_val)]
#         out = model(x_val)
#         preds = F.log_softmax(out, dim=1).argmax(dim=1)
#         total += y_val.size(0)
#         correct += (preds == y_val).sum().item()

#     acc = correct / total

#     if epoch % 5 == 0:
#         print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

#     if acc > best_acc:
#         trials = 0
#         best_acc = acc
#         torch.save(model.state_dict(), 'best.pth')
#         print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
#     else:
#         trials += 1
#         if trials >= patience:
#             print(f'Early stopping on epoch {epoch}')
#             break
