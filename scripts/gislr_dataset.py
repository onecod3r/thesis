"""
Dataset loader for the Kaggle Google - Isolated Sign Language Recognition
landmark parquet files.

Each sequence parquet has columns: frame, row_id, type, landmark_index, x, y, z
`type` is one of ['face', 'left_hand', 'pose', 'right_hand'].

We reassemble each sequence into a (T, 543, 3) array using the same global
landmark ordering as the local extraction pipeline:
    [0:468]   face        (468 points)
    [468:489] left_hand   (21 points)
    [489:522] pose        (33 points)
    [522:543] right_hand  (21 points)

Missing/undetected landmarks are NaN in the source data — kept as NaN through
normalization (so stats aren't biased by treating them as zero) and zero-filled
only at the very end.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TYPE_OFFSETS = {"face": 0, "left_hand": 468, "pose": 489, "right_hand": 522}
NUM_LANDMARKS = 543


def load_sequence(parquet_path):
    """Returns a (T, 543, 3) float32 array for one sequence."""
    df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])

    frames = np.sort(df["frame"].unique())
    frame_to_idx = {f: i for i, f in enumerate(frames)}
    T = len(frames)

    arr = np.full((T, NUM_LANDMARKS, 3), np.nan, dtype=np.float32)

    offsets = df["type"].map(TYPE_OFFSETS).values
    global_idx = offsets + df["landmark_index"].values
    frame_idx = df["frame"].map(frame_to_idx).values

    arr[frame_idx, global_idx, 0] = df["x"].values
    arr[frame_idx, global_idx, 1] = df["y"].values
    arr[frame_idx, global_idx, 2] = df["z"].values

    return arr


def build_label_map(train_csv_path, sign_map_json=None):
    """Uses sign_to_prediction_index_map.json if available (matches the
    competition's official label indices); otherwise builds one from the
    unique 'sign' values in train.csv, sorted for reproducibility."""
    if sign_map_json and os.path.exists(sign_map_json):
        with open(sign_map_json, "r") as f:
            return json.load(f)

    df = pd.read_csv(train_csv_path)
    signs = sorted(df["sign"].unique())
    return {sign: i for i, sign in enumerate(signs)}


class GISLRDataset(Dataset):
    def __init__(self, train_csv_path, data_root, sign_map_json=None,
                 max_len=128, use_z=False):
        self.df = pd.read_csv(train_csv_path)
        self.data_root = data_root
        self.max_len = max_len
        self.use_z = use_z
        self.label_map = build_label_map(train_csv_path, sign_map_json)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parquet_path = os.path.join(self.data_root, row["path"])
        arr = load_sequence(parquet_path)  # (T, 543, 3)

        if not self.use_z:
            arr = arr[:, :, :2]  # (T, 543, 2)

        # Normalize per-sample using nan-aware stats, then zero-fill.
        mean = np.nanmean(arr, axis=(0, 1), keepdims=True)
        std = np.nanstd(arr, axis=(0, 1), keepdims=True)
        std[std < 1e-6] = 1.0
        arr = (arr - mean) / std
        arr = np.nan_to_num(arr, nan=0.0)

        T = arr.shape[0]
        if T >= self.max_len:
            arr = arr[: self.max_len]
        else:
            pad = np.zeros((self.max_len - T, *arr.shape[1:]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        flat = arr.reshape(self.max_len, -1)  # (max_len, 543*coords)
        label = self.label_map[row["sign"]]

        return torch.from_numpy(flat).float(), torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self):
        return len(self.label_map)

    @property
    def feature_dim(self):
        coords = 3 if self.use_z else 2
        return NUM_LANDMARKS * coords
