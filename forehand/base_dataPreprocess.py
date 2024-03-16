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
        return len(self.openpose_files_df)

    def __getitem__(self, idx):
        openpose_df = pd.read_csv(self.openpose_files_df[idx]).drop(
            'frameNumber', axis=1)

        if self.transform:
            openpose_df = self.transform(openpose_df)

        ts = to_time_series([openpose_df.values])

        ts = torch.tensor(ts, dtype=torch.float32)
        if self.label_vtr_set is None:
            return ts[0]
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

        def OpenposePreprocess(df: pd.DataFrame, filename: str):
            isSave = False
            openpose_df = pd.read_csv(filename)

            openpose_df, isSaveTmp = GetTargetColumn(openpose_df)
            isSave = isSave or isSaveTmp

            openpose_df, isSaveTmp = CheckFrameNum(openpose_df)
            isSave = isSave or isSaveTmp

            openpose_df, isSaveTmp = CheckNormalize(openpose_df)
            isSave = isSave or isSaveTmp

            openpose_df, isSaveTmp = CheckNan(openpose_df, filename)
            isSave = isSave or isSaveTmp

            if isSave:
                openpose_df.to_csv(filename, index=False)
            return openpose_df

        def GetTargetColumn(df: pd.DataFrame):
            isSave = False
            target_column = "frameNumber	2DX_head	2DY_head	2DX_neck	2DY_neck	2DX_rshoulder	2DY_rshoulder	2DX_relbow	2DY_relbow	2DX_rhand	2DY_rhand	2DX_lshoulder	2DY_lshoulder	2DX_lelbow	2DY_lelbow	2DX_lhand	2DY_lhand	2DX_hip	2DY_hip	2DX_rhip	2DY_rhip	2DX_rknee	2DY_rknee	2DX_rfoot	2DY_rfoot	2DX_lhip	2DY_lhip	2DX_lknee	2DY_lknee	2DX_lfoot	2DY_lfoot	2DX_lheel	2DY_lheel	2DX_rheel	2DY_rheel".split(
                "\t")
            if not df.columns.equals(target_column):
                df = df[target_column]
                isSave = True
            return df, isSave

        def CheckNormalize(df: pd.DataFrame):
            isSave = False
            normalized_df = (df - df.mean()) / df.std()
            if not normalized_df.equals(df):
                isSave = True
                df = normalized_df
            return df, isSave

        def CheckFrameNum(df: pd.DataFrame):
            isSave = False
            if df['frameNumber'].shape[0] != self.FRAME:
                columns = df.columns
                ts = to_time_series_dataset([df.values])
                ts = TimeSeriesResampler(sz=self.FRAME).fit_transform(ts)
                df = pd.DataFrame(ts[0], columns=columns)
                isSave = True
            return df, isSave

        def CheckNan(df: pd.DataFrame, filename: str):
            isSave = False
            if df.isnull().values.any():
                print("NaN values found: " + filename)
                df.fillna(0, inplace=True)
                isSave = True
            return df, isSave

        if stage == 'predict':
            annotations_df = pd.read_csv(self.annotations_file)
            # Check the frame number
            for (i, row) in tqdm(annotations_df.iterrows(),
                                 total=annotations_df.shape[0]):
                OpenposePreprocess(annotations_df, row['openpose_files'])

            self.pred_dataset = MotionDataset(
                openpose_files_df=annotations_df['openpose_files'])

        else:
            annotations_df = pd.read_csv(self.annotations_file)

            # Check the frame number
            for (i, row) in tqdm(annotations_df.iterrows(),
                                 total=annotations_df.shape[0]):
                OpenposePreprocess(annotations_df, row['openpose_files'])

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
