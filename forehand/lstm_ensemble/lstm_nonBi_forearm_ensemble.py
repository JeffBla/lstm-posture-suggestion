import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import lightning as L
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataPreprocess_forearm_ensemble import MotionDataset, MotionDataModule
from lstm_nonBi_forearm_bin import LSTMClassifier as LSTMClassifier_bin
from lstm_nonBi_forearm import LSTMClassifier as LSTMClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--isFinalTest',
                    action='store_true',
                    help='does it start for final test')
parser.add_argument(
    "--annotations_file",
    type=str,
    default='',
    help=
    "the annotations file path(with openpose csv path, angle csv path, and label)"
)
parser.add_argument("--batch_size",
                    type=int,
                    default=8,
                    help="size of the batches")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=4,
    help="number of cpu threads to use during batch generation")
parser.add_argument("--binary_num_labels",
                    type=int,
                    default=2,
                    help="the number of label")
parser.add_argument("--multiclass_num_labels",
                    type=int,
                    default=3,
                    help="the number of label")
parser.add_argument("--input_dim",
                    type=int,
                    default=78,
                    help="the input dimension of the lstm")
parser.add_argument("--binary_hidden_dim",
                    type=int,
                    default=256,
                    help="the hidden dimension of the lstm")
parser.add_argument("--multiclass_hidden_dim",
                    type=int,
                    default=512,
                    help="the hidden dimension of the lstm")
parser.add_argument("--binary_layer_dim",
                    type=int,
                    default=3,
                    help="the layer dimension of the lstm")
parser.add_argument("--multiclass_layer_dim",
                    type=int,
                    default=5,
                    help="the layer dimension of the lstm")
parser.add_argument("--binary_output_dim", type=int, default=4)
parser.add_argument("--multiclass_output_dim", type=int, default=4)
parser.add_argument("--binary_prev_ckpt_path",
                    type=str,
                    default=None,
                    help="load previous binary classifier checkpoint")
parser.add_argument("--multiclass_prev_ckpt_path",
                    type=str,
                    default=None,
                    help="load previous multiclass classifier checkpoint")
opt = parser.parse_args()


class LSTMClassifierEnsemble(L.LightningModule):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, lstm_bin_classifier, lstm_classifier):
        super().__init__()
        self.lstm_bin_classifier = lstm_bin_classifier
        self.lstm_classifier = lstm_classifier

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.loss = None
        self.accuracy = MulticlassAccuracy(num_classes=opt.num_labels)
        self.recall = MulticlassRecall(num_classes=opt.num_labels)
        self.precision = MulticlassPrecision(num_classes=opt.num_labels)
        self.f1_score = MulticlassF1Score(num_classes=opt.num_labels)
        self.confusion_matrix = MulticlassConfusionMatrix(
            num_classes=opt.num_labels)
        self.finalTestPreds = None
        self.finalTestTargets = None
        self.dm = dm
        # Save hyperparameters
        self.save_hyperparameters({
            'batch_size':
            opt.batch_size,
            'n_cpu':
            opt.n_cpu,
            'binary_num_labels':
            opt.binary_num_labels,
            'multiclass_num_labels':
            opt.multiclass_num_labels,
            'input_dim':
            opt.input_dim,
            'binary_hidden_dim':
            opt.binary_hidden_dim,
            'binary_layer_dim':
            opt.binary_layer_dim,
            'multiclass_layer_dim':
            opt.multiclass_layer_dim,
            'binary_output_dim':
            opt.binary_output_dim,
            'multiclass_output_dim':
            opt.multiclass_output_dim,
        })

    def forward(self, x):
        binary_not = self.lstm_bin_classifier(x)
        bin_pred = self.sigmoid(binary_not).round()

        if bin_pred == 0:
            multiclass_out = self.lstm_classifier(x)
            multiclass_out = self.softmax(multiclass_out)
            return multiclass_out
        else:
            return binary_not

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = F.cross_entropy(y_hat, y)
        # self.log('final_test_loss',
        #          loss,
        #          on_step=True,
        #          on_epoch=True,
        #          prog_bar=True,
        #          logger=True)
        y = torch.max(y, 1)[1]
        y_hat = self.softmax(y_hat)
        if self.finalTestPreds is None:
            self.finalTestPreds = y_hat.clone().detach()
            self.finalTestTargets = y.clone().detach()
        else:
            self.finalTestPreds = torch.cat([self.finalTestPreds, y_hat])
            self.finalTestTargets = torch.cat([self.finalTestTargets, y])
        self.accuracy(y_hat, y)
        self.log('final_test_acc',
                 self.accuracy,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.recall(y_hat, y)
        self.log('final_test_recall',
                 self.recall,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.precision(y_hat, y)
        self.log('final_test_precision',
                 self.precision,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.f1_score(y_hat, y)
        self.log('final_test_f1_score',
                 self.f1_score,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return 1

    def on_test_epoch_end(self):
        confusion_matrix = self.confusion_matrix(self.finalTestPreds,
                                                 self.finalTestTargets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(),
                             index=range(opt.num_labels),
                             columns=range(opt.num_labels))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure("Confusion matrix", fig_, 0)

    def predict_step(self, batch, batch_idx):
        pred = self(batch)
        return pred

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
                                                    output_dim=opt.output_dim,
                                                    dm=dm)
        trainer = L.Trainer(accelerator='auto')
        pred = trainer.predict(model, dm)
        return pred


if __name__ == "__main__":

    if opt.isFinalTest:
        ################### Final Test #####################
        dm = MotionDataModule(annotations_file=opt.annotations_file,
                              batch_size=opt.batch_size,
                              n_cpu=opt.n_cpu)
        binary_model = LSTMClassifier_bin.load_from_checkpoint(
            opt.binary_prev_ckpt_path,
            input_dim=opt.input_dim,
            hidden_dim=opt.binary_hidden_dim,
            layer_dim=opt.binary_layer_dim,
            output_dim=opt.binary_output_dim,
            dm=dm)
        multiclass_model = LSTMClassifier.load_from_checkpoint(
            opt.multiclass_prev_ckpt_path,
            input_dim=opt.input_dim,
            hidden_dim=opt.multiclass_hidden_dim,
            layer_dim=opt.mulitclass_layer_dim,
            output_dim=opt.multiclass_output_dim,
            dm=dm)
        ensemble_model = LSTMClassifierEnsemble(binary_model, multiclass_model)

        trainer = L.Trainer(accelerator='auto')
        trainer.test(ensemble_model, dm)

    else:
        ################### Predict #####################
        dm = MotionDataModule(annotations_file=opt.annotations_file,
                              batch_size=opt.batch_size,
                              n_cpu=opt.n_cpu)
        model = LSTMClassifier.load_from_checkpoint(opt.prev_ckpt_path)
        trainer = L.Trainer(accelerator='auto')
        pred = trainer.predict(model, dm)
        print(pred)
