import os
import gc
import random
import time
from os.path import basename, dirname, join, exists

import json
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
from datetime import datetime
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler as Scaler

import warnings

warnings.filterwarnings(action="ignore")

BASE_DIR = "/ceph/csedu-scratch/course/IMC030_MLIP/data/tlvmc-parkinsons-freezing-gait-prediction"


@click.option("--dropout", type=float, default=0.0, help="dropout rate", required=False)
def main(dropout: float = 0.0):
    # Constants

    TRAIN_DIR = join(BASE_DIR, "train")
    TEST_DIR = join(BASE_DIR, "test")

    IS_PUBLIC = len(glob(join(TEST_DIR, "*/*.csv"))) == 2

    cfg = Config()

    # K-FOLD TDCS
    n1_sum = []
    n2_sum = []
    n3_sum = []
    count = []

    # Here I am using the metadata file available during training. Since the code will run again during submission, if
    # I used the usual file from the competition folder, it would have been updated with the test files too.
    metadata = pd.read_csv(BASE_DIR + "/tdcsfog_metadata.csv")

    for f in tqdm(metadata["Id"]):
        fpath = BASE_DIR + f"/train/tdcsfog/{f}.csv"
        df = pd.read_csv(fpath)

        n1_sum.append(np.sum(df["StartHesitation"]))
        n2_sum.append(np.sum(df["Turn"]))
        n3_sum.append(np.sum(df["Walking"]))
        count.append(len(df))

    print(f"32 files have positive values in all 3 classes")

    metadata["n1_sum"] = n1_sum
    metadata["n2_sum"] = n2_sum
    metadata["n3_sum"] = n3_sum
    metadata["count"] = count

    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(
        sgkf.split(X=metadata["Id"], y=[1] * len(metadata), groups=metadata["Subject"])
    ):
        print(f"Fold = {i}")
        train_ids = metadata.loc[train_index, "Id"]
        valid_ids = metadata.loc[valid_index, "Id"]

        print(
            f"Length of Train = {len(train_index)}, Length of Valid = {len(valid_index)}"
        )
        n1_sum = metadata.loc[train_index, "n1_sum"].sum()
        n2_sum = metadata.loc[train_index, "n2_sum"].sum()
        n3_sum = metadata.loc[train_index, "n3_sum"].sum()
        print(f"Train classes: {n1_sum:,}, {n2_sum:,}, {n3_sum:,}")

        n1_sum = metadata.loc[valid_index, "n1_sum"].sum()
        n2_sum = metadata.loc[valid_index, "n2_sum"].sum()
        n3_sum = metadata.loc[valid_index, "n3_sum"].sum()
        print(f"Valid classes: {n1_sum:,}, {n2_sum:,}, {n3_sum:,}")

    # FOLD 2 is the most well balanced
    # The actual train-test split (based on Fold 2)
    metadata = pd.read_csv(BASE_DIR + "/tdcsfog_metadata.csv")
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(
        sgkf.split(X=metadata["Id"], y=[1] * len(metadata), groups=metadata["Subject"])
    ):
        if i != 2:
            continue
        print(f"Fold = {i}")
        train_ids = metadata.loc[train_index, "Id"]
        valid_ids = metadata.loc[valid_index, "Id"]
        print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")

        if i == 2:
            break

    train_fpaths_tdcs = [BASE_DIR + f"/train/tdcsfog/{_id}.csv" for _id in train_ids]
    valid_fpaths_tdcs = [BASE_DIR + f"/train/tdcsfog/{_id}.csv" for _id in valid_ids]

    # METADATA

    metadata = pd.read_csv(BASE_DIR + "/tdcsfog_metadata.csv")
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(
        sgkf.split(X=metadata["Id"], y=[1] * len(metadata), groups=metadata["Subject"])
    ):
        if i != 2:
            continue
        print(f"Fold = {i}")
        train_ids = metadata.loc[train_index, "Id"]
        valid_ids = metadata.loc[valid_index, "Id"]
        print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")

        if i == 2:
            break

    train_fpaths_tdcs = [BASE_DIR + f"/train/tdcsfog/{_id}.csv" for _id in train_ids]
    valid_fpaths_tdcs = [BASE_DIR + f"/train/tdcsfog/{_id}.csv" for _id in valid_ids]

    # kfold - DEFOG
    n1_sum = []
    n2_sum = []
    n3_sum = []
    count = []

    # Here I am using the metadata file available during training. Since the code will run again during submission, if
    # I used the usual file from the competition folder, it would have been updated with the test files too.
    metadata = pd.read_csv(BASE_DIR + "/defog_metadata.csv")
    metadata["n1_sum"] = 0
    metadata["n2_sum"] = 0
    metadata["n3_sum"] = 0
    metadata["count"] = 0

    for f in tqdm(metadata["Id"]):
        fpath = BASE_DIR + f"/train/defog/{f}.csv"
        if os.path.exists(fpath) == False:
            continue

        df = pd.read_csv(fpath)
        metadata.loc[metadata["Id"] == f, "n1_sum"] = np.sum(df["StartHesitation"])
        metadata.loc[metadata["Id"] == f, "n2_sum"] = np.sum(df["Turn"])
        metadata.loc[metadata["Id"] == f, "n3_sum"] = np.sum(df["Walking"])
        metadata.loc[metadata["Id"] == f, "count"] = len(df)

    metadata = metadata[metadata["count"] > 0].reset_index()

    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(
        sgkf.split(X=metadata["Id"], y=[1] * len(metadata), groups=metadata["Subject"])
    ):
        print(f"Fold = {i}")
        train_ids = metadata.loc[train_index, "Id"]
        valid_ids = metadata.loc[valid_index, "Id"]

        print(
            f"Length of Train = {len(train_index)}, Length of Valid = {len(valid_index)}"
        )
        n1_sum = metadata.loc[train_index, "n1_sum"].sum()
        n2_sum = metadata.loc[train_index, "n2_sum"].sum()
        n3_sum = metadata.loc[train_index, "n3_sum"].sum()
        print(f"Train classes: {n1_sum:,}, {n2_sum:,}, {n3_sum:,}")

        n1_sum = metadata.loc[valid_index, "n1_sum"].sum()
        n2_sum = metadata.loc[valid_index, "n2_sum"].sum()
        n3_sum = metadata.loc[valid_index, "n3_sum"].sum()
        print(f"Valid classes: {n1_sum:,}, {n2_sum:,}, {n3_sum:,}")

    # FOLD 2 is the most well balanced
    # The actual train-test split (based on Fold 2)

    sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for i, (train_index, valid_index) in enumerate(
        sgkf.split(X=metadata["Id"], y=[1] * len(metadata), groups=metadata["Subject"])
    ):
        if i != 1:
            continue
        print(f"Fold = {i}")
        train_ids = metadata.loc[train_index, "Id"]
        valid_ids = metadata.loc[valid_index, "Id"]
        print(f"Length of Train = {len(train_ids)}, Length of Valid = {len(valid_ids)}")

        if i == 2:
            break

    train_fpaths_de = [BASE_DIR + f"/train/defog/{_id}.csv" for _id in train_ids]
    valid_fpaths_de = [BASE_DIR + f"/train/defog/{_id}.csv" for _id in valid_ids]

    train_fpaths_de = [BASE_DIR + f"/train/defog/{_id}.csv" for _id in train_ids]
    valid_fpaths_de = [BASE_DIR + f"/train/defog/{_id}.csv" for _id in valid_ids]

    train_fpaths = [(f, "de") for f in train_fpaths_de] + [
        (f, "tdcs") for f in train_fpaths_tdcs
    ]
    valid_fpaths = [(f, "de") for f in valid_fpaths_de] + [
        (f, "tdcs") for f in valid_fpaths_tdcs
    ]

    # SCALING

    if cfg.norm:
        # TODO: Try also folderwise
        tdcsfog_fit_values, other_fit_values = [], []
        for df_path in tqdm(glob(join(TRAIN_DIR, "*/*.csv"))):
            df = pd.read_csv(
                df_path, index_col="Time", usecols=["Time", *cfg.norm_list]
            )
            if basename(dirname(df_path)) == "tdcsfog":
                tdcsfog_fit_values.append(df[cfg.norm_list])
            else:
                other_fit_values.append(df[cfg.norm_list])

        print(
            f"Fitting {len(tdcsfog_fit_values)} tdcsfog values and {len(other_fit_values)} other values."
        )
        tdcsfog_scaler = Scaler().fit(pd.concat(tdcsfog_fit_values))
        other_scaler = Scaler().fit(pd.concat(other_fit_values))

    gc.collect()

    # print(f"Number of parameters in model - {count_parameters(model):,}")

    train_dataset = FOGDataset(
        train_fpaths,
        cfg,
        split="train",
        tdcsfog_scaler=tdcsfog_scaler,
        other_scaler=other_scaler,
    )
    valid_dataset = FOGDataset(
        valid_fpaths,
        cfg,
        split="valid",
        tdcsfog_scaler=tdcsfog_scaler,
        other_scaler=other_scaler,
    )
    print(
        f"lengths of datasets: train - {len(train_dataset)}, valid - {len(valid_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=5, shuffle=True
    )
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=5)

    # Instantiate the LSTM model
    input_size = 3  # Number of measurements at each time point
    hidden_size = 64  # Number of LSTM units or hidden dimensions
    num_layers = 2  # Number of LSTM layers
    output_size = 3  # Size of the output (binary classification)

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(
        cfg.device
    )

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    max_score = 0.0
    dt = datetime.now().strftime("%Y%m%d%H:%M")

    model_name = "lstm_step_lr_{dt}"
    print("=" * 50)
    model.train()
    for epoch in range(cfg.num_epochs):
        print(f"Epoch: {epoch}")
        train_one_epoch(model, train_loader, optimizer, criterion)
        score = validation_one_epoch(model, valid_loader, criterion)

        if score > max_score:
            max_score = score
            torch.save(model.state_dict(), f"best_model-{model_name}.h5")
            print("Saving Model ...")
        scheduler.step()
        print("=" * 50)

    gc.collect()


class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.0
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)  # Apply dropout regularization
        out = self.fc(out[:, -1, :])  # Take the last time step output
        out = self.sigmoid(out)

        return out


def train_one_epoch(model, loader, optimizer, criterion):
    loss_sum = 0.0
    scaler = GradScaler()

    model.train()
    for x, y, t in tqdm(loader):
        x = x.to(cfg.device).float()
        y = y.to(cfg.device).float()
        t = t.to(cfg.device).float()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss = torch.mean(loss * t.unsqueeze(-1), dim=1)

        t_sum = torch.sum(t)
        if t_sum > 0:
            loss = torch.sum(loss) / t_sum
        else:
            loss = torch.sum(loss) * 0.0

        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        loss_sum += loss.item()

    print(f"Train Loss: {(loss_sum/len(loader)):.04f}")


def validation_one_epoch(model, loader, criterion):
    loss_sum = 0.0
    y_true_epoch = []
    y_pred_epoch = []
    t_valid_epoch = []

    model.eval()
    for x, y, t in tqdm(loader):
        x = x.to(cfg.device).float()
        y = y.to(cfg.device).float()
        t = t.to(cfg.device).float()

        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss = torch.mean(loss * t.unsqueeze(-1), dim=1)

            t_sum = torch.sum(t)
            if t_sum > 0:
                loss = torch.sum(loss) / t_sum
            else:
                loss = torch.sum(loss) * 0.0

        loss_sum += loss.item()
        y_true_epoch.append(y.cpu().numpy())
        y_pred_epoch.append(y_pred.cpu().numpy())
        t_valid_epoch.append(t.cpu().numpy())

    y_true_epoch = np.concatenate(y_true_epoch, axis=0)
    y_pred_epoch = np.concatenate(y_pred_epoch, axis=0)

    t_valid_epoch = np.concatenate(t_valid_epoch, axis=0)
    y_true_epoch = y_true_epoch[t_valid_epoch > 0, :]
    y_pred_epoch = y_pred_epoch[t_valid_epoch > 0, :]

    scores = [
        average_precision_score(y_true_epoch[:, i], y_pred_epoch[:, i])
        for i in range(3)
    ]
    mean_score = np.mean(scores)
    print(
        f"Validation Loss: {(loss_sum/len(loader)):.04f}, Validation Score: {mean_score:.03f}, ClassWise: {scores[0]:.03f},{scores[1]:.03f},{scores[2]:.03f}"
    )

    return mean_score


# DATASET
class FOGDataset(Dataset):
    def __init__(
        self,
        fpaths,
        cfg,
        scale=9.806,
        split="train",
        tdcsfog_scaler=None,
        other_scaler=None,
    ):
        super(FOGDataset, self).__init__()
        tm = time.time()
        self.split = split
        self.scale = scale
        self.cfg = cfg
        self.tdcsfog_scaler = tdcsfog_scaler
        self.other_scaler = other_scaler
        self.fpaths = fpaths
        self.dfs = [self.read(f[0], f[1]) for f in fpaths]
        self.f_ids = [os.path.basename(f[0])[:-4] for f in self.fpaths]

        self.end_indices = []
        self.shapes = []
        _length = 0
        for df in self.dfs:
            self.shapes.append(df.shape[0])
            _length += df.shape[0]
            self.end_indices.append(_length)

        self.dfs = np.concatenate(self.dfs, axis=0).astype(np.float16)
        self.length = self.dfs.shape[0]

        shape1 = self.dfs.shape[1]

        self.dfs = np.concatenate(
            [
                np.zeros((cfg.wx * cfg.window_past, shape1)),
                self.dfs,
                np.zeros((cfg.wx * cfg.window_future, shape1)),
            ],
            axis=0,
        )
        print(f"Dataset initialized in {time.time() - tm} secs!")
        gc.collect()

    def read(self, f, _type):
        df = pd.read_csv(f)
        if self.split == "test":
            return np.array(df)

        if _type == "tdcs":
            df["Valid"] = 1
            df["Task"] = 1
            df["tdcs"] = 1
        else:
            df["tdcs"] = 0

        if self.cfg.norm:
            if _type != "tdcs":
                df[cfg.norm_list] = self.tdcsfog_scaler.transform(df[cfg.norm_list])
            else:
                df[cfg.norm_list] = self.other_scaler.transform(df[cfg.norm_list])

        #         df['Time_frac'] = (df.Time/df.Time.max()).astype('float16')

        return np.array(df)

    def __getitem__(self, index):
        if self.split == "train":
            row_idx = random.randint(0, self.length - 1) + cfg.wx * cfg.window_past
        elif self.split == "test":
            for i, e in enumerate(self.end_indices):
                if index >= e:
                    continue
                df_idx = i
                break

            row_idx_true = self.shapes[df_idx] - (self.end_indices[df_idx] - index)
            _id = self.f_ids[df_idx] + "_" + str(row_idx_true)
            row_idx = index + cfg.wx * cfg.window_past
        else:
            row_idx = index + cfg.wx * cfg.window_past

        # scale = 9.806 if self.dfs[row_idx, -1] == 1 else 1.0
        x = self.dfs[
            row_idx - cfg.wx * cfg.window_past : row_idx + cfg.wx * cfg.window_future,
            1:4,
        ]
        x = x[:: cfg.wx, :][::-1, :]
        x = torch.tensor(x.astype("float"))  # /scale

        t = self.dfs[row_idx, -3] * self.dfs[row_idx, -2]

        if self.split == "test":
            return _id, x, t

        y = self.dfs[row_idx, 4:7].astype("float")
        y = torch.tensor(y)

        return x, y, t

    def __len__(self):
        # return self.length
        if self.split == "train":
            return 5_000_000
        return self.length


def format_time(seconds):
    if seconds > 3600:
        return f"{seconds/3600:.2f} hrs"
    if seconds > 60:
        return f"{seconds/60:.2f} mins"
    return f"{seconds:.2f} secs"


class Config:
    train_dir1 = BASE_DIR + "/train/defog"
    train_dir2 = BASE_DIR + "/train/tdcsfog"

    batch_size = 512
    window_size = 32
    window_future = 8
    window_past = window_size - window_future

    wx = 8

    embed_dim = 128

    lr = 0.001
    num_epochs = 40
    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm = True
    #     feature_list = ['Time_frac','AccV', 'AccML', 'AccAP']
    feature_list = ["AccV", "AccML", "AccAP"]
    label_list = ["StartHesitation", "Turn", "Walking"]
    norm_list = ["AccV", "AccML", "AccAP"]

    n_features = len(feature_list)
    n_labels = len(label_list)


cfg = Config()

if __name__ == "__main__":
    main()
