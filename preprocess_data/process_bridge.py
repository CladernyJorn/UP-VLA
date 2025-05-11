import os
import torch.multiprocessing
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tqdm
import json
import cv2


def decode_inst(inst):
    """Utility to decode encoded language instruction"""
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")


def preprocess_dataset(origin_dataset_path, episode_num=10):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    root = "./bridge_processed"
    builder = tfds.builder_from_directory(origin_dataset_path)
    episode_ds = builder.as_dataset(split='train')
    print(episode_ds.element_spec)
    os.makedirs(root, exist_ok=True)
    dataset_info = []
    print('------processing------')
    for i, episode in enumerate(tqdm.tqdm(iter(episode_ds.take(episode_num)))):
        episode_path = os.path.join(root, f"episode{i:07}")
        os.makedirs(episode_path, exist_ok=True)
        episode_info = {'instruction': "", 'frames': []}
        for j, step in enumerate(episode['steps'].as_numpy_iterator()):
            frame = step['observation']['rgb']  # ndarray
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(episode_path, f"frame{j:03}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_info = {'dir': f"episode{i:07}/frame{j:03}.jpg", 'action': step['action'].tolist()}
            episode_info['frames'].append(frame_info)
            if j == 0:
                episode_info['instruction'] = decode_inst(step['observation']['instruction'])
        dataset_info.append(episode_info)
    with open(os.path.join(root, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)


preprocess_dataset("/mnt/bridge/")  # change the path to your own raw bridge dataset

# python process_bridge.py
