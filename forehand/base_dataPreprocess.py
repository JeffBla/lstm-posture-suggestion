from tqdm import tqdm
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

LABEL = "waist_twist_correct	waist_twist_aggressively	waist_no_twist	forhand_correct	forehand_wave_aggressively	forehand_no_wave".split(
    '\t')


class MotionDataset(Dataset):

    def __init__(self, openpose_files_df, label_vtr_set=None, transform=None):
        self.openpose_files_df = openpose_files_df
        self.label_vtr_set = label_vtr_set
        self.transform = transform

    def __len__(self):
        return len(self.label_vtr_set)

    def __getitem__(self, idx):
        openpose_df = pd.read_csv(self.openpose_files_df[idx]).drop(
            'frameNumber', axis=1)

        if self.transform:
            openpose_df = self.transform(openpose_df)

        ts = to_time_series([openpose_df.values])

        ts = torch.tensor(ts, dtype=torch.float32)
        if self.label_vtr_set is None:
            return ts
        else:
            label_vtr = torch.tensor(self.label_vtr_set.iloc[idx].values,
                                     dtype=torch.float32)
            return ts[0], label_vtr


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

        def CheckNormalize(df: pd.DataFrame, filename: str):
            normalized_df = (df - df.mean()) / df.std()
            if not normalized_df.equals(df):
                normalized_df.to_csv(filename, index=False)
                return normalized_df
            return df

        def CheckFrameNum(df: pd.DataFrame, filename: str):
            if df['frameNumber'].shape[0] != self.FRAME:
                columns = df.columns
                ts = to_time_series_dataset([df.values])
                ts = TimeSeriesResampler(sz=self.FRAME).fit_transform(ts)
                df = pd.DataFrame(ts[0], columns=columns)
                df.to_csv(filename, index=False)
                return df
            else:
                return df

        def CheckNan(df: pd.DataFrame, filename: str):
            if df.isnull().values.any():
                print("NaN values found: " + filename)
                # df.fillna(0, inplace=True)
                # df.to_csv(filename, index=False)
                return df
            else:
                return df

        if stage == 'predict':
            annotations_df = pd.read_csv(self.annotations_file)
            # Check the frame number
            for (i, row) in annotations_df.iterrows():
                openpose_df = pd.read_csv(row['openpose_files'])
                CheckFrameNum(openpose_df)
                CheckNormalize(openpose_df)

            self.pred_dataset = MotionDataset(
                openpose_files_df=annotations_df['openpose_files'])

        else:
            annotations_df = pd.read_csv(self.annotations_file)

            # Check the frame number
            for (i, row) in tqdm(annotations_df.iterrows(),
                                 total=annotations_df.shape[0]):
                openpose_df = pd.read_csv(row['openpose_files'])
                openpose_df = CheckFrameNum(openpose_df, row['openpose_files'])
                openpose_df = CheckNormalize(openpose_df,
                                             row['openpose_files'])
                openpose_df = CheckNan(openpose_df, row['openpose_files'])

            self.dataset = MotionDataset(
                openpose_files_df=annotations_df['openpose_files'],
                label_vtr_set=annotations_df[LABEL])
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
