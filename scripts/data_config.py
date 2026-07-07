"""
Resolves dataset paths using kagglehub's cached competition download, so you
don't need to hand-pass --train_csv/--data_root every run.
"""
import os
import kagglehub


def get_dataset_paths(competition="asl-signs"):
    """Returns (data_root, train_csv_path, sign_map_json_path).
    kagglehub caches the download, so repeated calls are cheap after the
    first run — this doesn't re-download every time you call it."""
    data_root = kagglehub.competition_download(competition)
    train_csv = os.path.join(data_root, "train.csv")
    sign_map_json = os.path.join(data_root, "sign_to_prediction_index_map.json")

    if not os.path.exists(sign_map_json):
        sign_map_json = None  # dataset.py falls back to building the map from train.csv

    return data_root, train_csv, sign_map_json
