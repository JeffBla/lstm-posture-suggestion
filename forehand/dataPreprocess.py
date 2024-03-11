import os
import json
import numpy as np
import pandas as pd

from tslearn.utils import to_time_series_dataset, to_time_series
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMeanVariance

import torch
from torch.utils.data import DataLoader, Dataset, random_split

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


class MotionDataset(Dataset):

    def __init__(self,
                 openpose_files_df,
                 angle_files_df,
                 label_vtr_set=None,
                 transform=None):
        self.openpose_files_df = openpose_files_df
        self.angle_files_df = angle_files_df
        self.label_vtr_set = label_vtr_set
        self.transform = transform

    def __len__(self):
        return len(self.label_vtr_set)

    def __getitem__(self, idx):
        openpose_df = pd.read_csv(
            self.openpose_files_df[idx]).drop(columns=['frame'])
        angle_df = pd.read_csv(self.angle_df_set[idx]).drop(columns=['frame'])

        if self.transform:
            openpose_df, angle_df = self.transform(openpose_df, angle_df)

        ts = to_time_series(
            np.concatenate([openpose_df.values, angle_df.values], axis=1))

        if self.label_vtr_set is None:
            return ts
        else:
            return ts, self.label_vtr_set[idx]


class MotionDataModule(L.LightningDataModule):

    def __init__(self,
                 batch_size: int = 32,
                 annotations_file: str = None,
                 n_cpu: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.annotations_file = annotations_file
        self.n_cpu = n_cpu
        self.train_test_spilt = 0.7
        # the len of ts should be the same in lstm
        self.FRAME = 100

        self.dataset = None
        self.pred_dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage):

        def CheckFrameNum(df: pd.DataFrame, filename: str):
            scale = TimeSeriesScalerMeanVariance(mu=0., std=1.)

            if df['frame'].shape[0] != self.FRAME:
                ts = to_time_series(df.values)
                ts = TimeSeriesResampler(sz=self.FRAME).fit_transform(ts)
                ts = scale.fit_transform(ts)
                df = pd.DataFrame(ts[0])
                df.to_csv(filename, index=False)
                return df
            else:
                return df

        if stage == 'predict':
            annotations_df = pd.read_csv(self.annotations_file)
            # Check the frame number
            for (i, row) in annotations_df.iterrows():
                openpose_df = pd.read_csv(row['openpose_files'])
                angle_df = pd.read_csv(row['angle_files'])
                CheckFrameNum(openpose_df)
                CheckFrameNum(angle_df)

            self.pred_dataset = MotionDataset(
                openpose_files_df=annotations_df['openpose_files'],
                angle_files_df=annotations_df['angle_files'])

        else:
            annotations_df = pd.read_csv(self.annotations_file)

            # Check the frame number
            for (i, row) in annotations_df.iterrows():
                openpose_df = pd.read_csv(row['openpose_files'])
                angle_df = pd.read_csv(row['angle_files'])
                CheckFrameNum(openpose_df)
                CheckFrameNum(angle_df)

            self.dataset = MotionDataset(
                openpose_files_df=annotations_df['openpose_files'],
                angle_files_df=annotations_df['angle_files'],
                label_vtr_set=annotations_df['label_vtr_set'])
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.test_dataset = random_split(
                self.dataset,
                [self.train_test_spilt, 1 - self.train_test_spilt], generator)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_cpu)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.pred_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu)
