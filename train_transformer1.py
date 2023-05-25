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


import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler as Scaler

import warnings

warnings.filterwarnings(action="ignore")

BASE_DIR = "/ceph/csedu-scratch/course/IMC030_MLIP/data/tlvmc-parkinsons-freezing-gait-prediction"


def main():
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

    model = Transformer(
        input_shape=(cfg.window_size, cfg.n_features),
        num_transformer_blocks=cfg.num_transformer_blocks,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        mlp_units=cfg.mlp_units,
        dropout=cfg.model_dropout,
        mlp_dropout=cfg.mlp_dropout,
    ).to(cfg.device)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.08)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none").to(cfg.device)
    # sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

    max_score = 0.0

    print("=" * 50)
    for epoch in range(cfg.num_epochs):
        print(f"Epoch: {epoch}")
        train_one_epoch(model, train_loader, optimizer, criterion)
        score = validation_one_epoch(model, valid_loader, criterion)
        scheduler.step()

        if score > max_score:
            max_score = score
            torch.save(model.state_dict(), "best_model_state.h5")
            print("Saving Model ...")

        print("=" * 50)

    gc.collect()


class Encoder(nn.Module):
    def __init__(self, input_shape, num_heads_mha, p_dropout, ff_dim):
        super(Encoder, self).__init__()
        seq_length = input_shape[0]
        n_feats = input_shape[-1]
        self.layernorm_mha = nn.LayerNorm(normalized_shape=n_feats, eps=1e-6)
        self.dropout_mha = nn.Dropout(p_dropout)
        self.mha = nn.MultiheadAttention(
            n_feats, num_heads_mha, kdim=n_feats, dropout=p_dropout
        )
        self.layernorm_0 = nn.LayerNorm(normalized_shape=n_feats, eps=1e-6)
        self.conv1d_0 = nn.Conv1d(n_feats, ff_dim, kernel_size=1)
        self.act_0 = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.conv1d_1 = nn.Conv1d(ff_dim, n_feats, kernel_size=1)

    def forward(self, inputs):
        residual = inputs.clone()
        #         print(f"input shape to encoder forward: {inputs.shape}")
        x = self.layernorm_mha(inputs)
        #         print(f"shape after layernorm: {x.shape}")
        x = self.mha(query=x, key=x, value=x)[0]
        #         print(f"shape after mha: {x.shape}")
        x = self.dropout_mha(x)
        x += residual

        residual_x = x.clone()
        y = self.layernorm_0(x).transpose(1, 2)
        #         print(f"shape after layernorm: {x.shape}")
        y = self.conv1d_0(y)
        y = self.act_0(y)
        y = self.dropout(y)
        y = self.conv1d_1(y).transpose(1, 2)

        return residual_x + y


def convert_channels_last_to_first(x):
    return x.permute(0, 2, 1)


# Define the GlobalAveragePooling1D layer
class GlobalAveragePooling1D(nn.Module):
    def forward(self, x):
        # Convert 'channels_last' to 'channels_first' format
        x = convert_channels_last_to_first(x)

        # Apply AdaptiveAvgPool1d to perform global average pooling
        x = nn.AdaptiveAvgPool1d(1)(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        return x


class Transformer(nn.Module):
    """
    NOTE: Expects the input to be normalized.
    """

    def __init__(
        self,
        input_shape,
        num_transformer_blocks,
        num_heads,
        ff_dim,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
    ):
        super(Transformer, self).__init__()
        self.n_classes = input_shape[-1]
        self.seq_len = input_shape[0]
        self.mlp_units = mlp_units
        self.e_layers = nn.ModuleList(
            [
                Encoder(
                    input_shape=input_shape,
                    num_heads_mha=num_heads,
                    p_dropout=dropout,
                    ff_dim=ff_dim,
                )
                for i in range(num_transformer_blocks)
            ]
        )

        self.avgpooling = GlobalAveragePooling1D()
        self.mlp_layers = []
        prev_dim = self.n_classes
        for dim in self.mlp_units:
            self.mlp_layers.append(nn.Linear(prev_dim, dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(mlp_dropout))
            prev_dim = dim
        self.mlp_layers_seq = nn.Sequential(*self.mlp_layers)
        self.dense_0 = nn.Linear(dim, self.n_classes)

    def forward(self, inputs):
        x = inputs
        for layer in self.e_layers:
            x = layer(x)
        x = self.avgpooling(x)
        x = self.mlp_layers_seq(x)
        x = self.dense_0(x)
        return x


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

    batch_size = 1024
    window_size = 32
    window_future = 8
    window_past = window_size - window_future

    wx = 8

    model_dropout = 0.2
    embed_dim = 128

    model_dropout = 0.1
    num_heads = 3
    ff_dim = 4
    num_transformer_blocks = 4
    mlp_units = [128]
    mlp_dropout = model_dropout * 2

    lr = 0.001
    num_epochs = 8
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
