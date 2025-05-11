import os
import torch.multiprocessing
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tqdm
import json
import cv2
from PIL import Image
import shutil


def preprocess_dataset(origin_dataset_path):
    root = "./calvin_processed" + origin_dataset_path.split("/")[-1]
    lang_info = np.load(
        os.path.join(origin_dataset_path, 'lang_annotations', 'auto_lang_ann.npy'), allow_pickle=True).item()
    texts = lang_info['language']['ann']
    episode_start_end = lang_info['info']['indx']
    os.makedirs(root, exist_ok=True)
    dataset_info = []
    for i, ((episode_start, episode_end), instruction) in tqdm.tqdm(enumerate(zip(episode_start_end, texts))):
        # if i < 10:
        #     episode_path = os.path.join(root, f"episode{i:07}")
        #     os.makedirs(episode_path, exist_ok=True)
        episode_info = {'instruction': instruction, 'frames': []}
        for j, frame_idx in enumerate(range(episode_start, episode_end + 1)):
            file_path = os.path.join(origin_dataset_path, 'episode_%07d.npz' % frame_idx)
            data = np.load(file_path, allow_pickle=True)
            # if i < 10:
            #     frame_path1 = os.path.join(episode_path, f"frame_gripper{j:03}.jpg")
            #     frame_path3 = os.path.join(episode_path, f"frame_static{j:03}.jpg")
            #     cv2.imwrite(frame_path1, data['rgb_gripper'][:, :, ::-1])
            #     cv2.imwrite(frame_path3, data['rgb_static'][:, :, ::-1])
            frame_info = {
                'dir': file_path,
                'abs_action': data['actions'].tolist(),
                'rel_action': data['rel_actions'].tolist()
            }
            episode_info['frames'].append(frame_info)
        dataset_info.append(episode_info)
    with open(os.path.join(root, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)


# preprocess_dataset("/localssd/hyc_data/calvin/task_ABC_D/validation")
preprocess_dataset("/mnt/calvin/task_ABC_D/validation")  # change the path to your own raw calvin abcd dataset
preprocess_dataset("/mnt/calvin/task_ABC_D/training")

# python process_calvin.py
