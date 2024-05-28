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

LABEL = "forhand_correct	forehand_wave_too_hard	forehand_wave_too_small	forehand_wave_wrong".split(
    '\t')


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
        return len(self.openpose_files_df)

    def __getitem__(self, idx):
        openpose_df = pd.read_csv(self.openpose_files_df[idx]).drop(
            'frameNumber', axis=1)
        angle_df = pd.read_csv(self.angle_files_df[idx]).drop(
            columns=['frameNumber'], axis=1)

        if self.transform:
            openpose_df, angle_df = self.transform(openpose_df, angle_df)

        ts = to_time_series(
            np.concatenate([openpose_df.values, angle_df.values],
                           axis=1)[np.newaxis, :, :])

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

        def DataPreprocess(df: pd.DataFrame, filename: str, isAngle=False):
            isSave = False

            if df.isna().any().any() or df.isin([np.inf, -np.inf]).any().any():
                df.fillna(0, inplace=True)
                df[df == np.inf] = 0
                df[df == -np.inf] = 0
                isSave = True

            df, isSaveTmp = CheckColumnsName_FrameNum(df)

            df, isSaveTmp = GetTargetColumn(df, isAngle)
            isSave = isSave or isSaveTmp

            df, isSaveTmp = CheckFrameNum(df)
            isSave = isSave or isSaveTmp

            df, isSaveTmp = CheckNormalize(df)
            isSave = isSave or isSaveTmp

            df, isSaveTmp = CheckNan(df, filename)
            isSave = isSave or isSaveTmp

            if isSave:
                df.to_csv(filename, index=False)
            return df

        def GetTargetColumn(df: pd.DataFrame, isAngle):
            isSave = False
            if isAngle:
                # Columns with constant values are no use for classification
                # https://stackoverflow.com/questions/36486120/normalisation-with-a-zero-in-the-standard-deviation
                target_column = "frameNumber	hip_x	hip_y	hip_z	hip_abdomen	hip_lButtock	hip_rButtock	rShldr_x	rShldr_y	rShldr_z	rShldr_rForeArm	rForeArm_x	rForeArm_y	rForeArm_z	rForeArm_rHand	rHand_x	rHand_y	rHand_z	lShldr_x	lShldr_y	lShldr_z	lShldr_lForeArm	lForeArm_x	lForeArm_y	lForeArm_z	lForeArm_lHand	lHand_x	lHand_y	lHand_z	rThigh_x	rThigh_y	rThigh_z	rThigh_rShin	rShin_x	rShin_y	rShin_z	rShin_rFoot	lThigh_x	lThigh_y	lThigh_z	lThigh_lShin	lShin_x	lShin_y	lShin_z	lShin_lFoot".split(
                    "\t")
                if not df.columns.equals(target_column):
                    df = df[target_column]
                    isSave = True
            else:
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

        def CheckColumnsName_FrameNum(df: pd.DataFrame):
            isSave = False
            if df.columns[0] != 'frameNumber':
                df.rename(columns={df.columns[0]: 'frameNumber'}, inplace=True)
                isSave = True
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

        annotations_df = pd.read_csv(self.annotations_file)
        if stage == 'predict' or stage == 'test':
            for (i, row) in tqdm(annotations_df.iterrows(),
                                 total=annotations_df.shape[0]):
                openpose_df = pd.read_csv(row['openpose_files'])
                angle_df = pd.read_csv(row['angle_files'])

                DataPreprocess(openpose_df, row['openpose_files'])
                DataPreprocess(angle_df, row['angle_files'], True)

            if stage == 'predict':
                self.pred_dataset = MotionDataset(
                    openpose_files_df=annotations_df['openpose_files'],
                    angle_files_df=annotations_df['angle_files'])
            else:
                self.test_dataset = MotionDataset(
                    openpose_files_df=annotations_df['openpose_files'],
                    angle_files_df=annotations_df['angle_files'],
                    label_vtr_set=annotations_df[LABEL])

        else:  # "fit stage"
            for (i, row) in tqdm(annotations_df.iterrows(),
                                 total=annotations_df.shape[0]):
                openpose_df = pd.read_csv(row['openpose_files'])
                angle_df = pd.read_csv(row['angle_files'])

                DataPreprocess(openpose_df, row['openpose_files'])
                DataPreprocess(angle_df, row['angle_files'], True)

            self.dataset = MotionDataset(
                openpose_files_df=annotations_df['openpose_files'],
                angle_files_df=annotations_df['angle_files'],
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
