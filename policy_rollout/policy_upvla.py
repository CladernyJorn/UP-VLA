import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
import sys
import torch
from PIL import Image

sys.path.insert(0, "./UP-VLA")
from models import Upvla, MAGVITv2, CLIPVisionTower
from training.prompting_utils import UniversalPrompting_w_action, \
    create_attention_mask_predict_next_for_future_prediction
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
from transformers import CLIPImageProcessor
from llava.llava import conversation as conversation_lib

conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
# SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
#                 "The assistant gives helpful, detailed, and polite answers to the user's questions."
# SYSTEM_PROMPT_LEN = 28
SYSTEM_PROMPT = ""
SYSTEM_PROMPT_LEN = 0
# from open3d.examples.visualization.to_mitsuba import dataset

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    BinaryRewardClassifierWrapper,
    ZeroRewardWrapper,
)
from supervised_roll_out.real_embodiment_roll_out import supervised_agent
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
# from mdt.deploy_panda import PandaAgent

from serl_launcher.wrappers.chunking import ChunkingWrapper
import argparse
import cv2
from matplotlib import pyplot as plt
import time


def save_img(transitions, file_path, traj_length=100):
    import json
    data_info = []
    for i, transition in enumerate(transitions):
        traj_id = int(np.floor(i / traj_length))
        step_id = int(i % traj_length)
        traj_path = f"{file_path}/episode{traj_id:04}"
        os.makedirs(traj_path, exist_ok=True)
        obs = transition["observations"]
        state = obs["state"].tolist()
        action = transition["actions"].tolist()
        data_info.append({"idx": i, "episode": traj_id, "step": step_id, "state": state, "action": action})
        img_names = ["wrist_1", "wrist_2"]
        for name in img_names:
            if name in obs.keys():
                # print(obs[name].shape)
                img = obs[name].squeeze(0)
                img_path = f"{traj_path}/color_{name}_{step_id:04}.jpg"
                depth_path = f"{traj_path}/depth_{name}_{step_id:04}.npy"
                img_rgb = np.array(img[..., :3], dtype=np.uint8)
                # print(img_rgb[:5,:5])
                depth = img[..., 3]
                depth = cv2.resize(depth, (128, 128))
                plt.imsave(img_path, img_rgb)
                np.save(depth_path, depth)
    with open(f"{file_path}/data_info.json", "w") as f:
        json.dump(data_info, f)
    print(f"saved {success_needed} demos and {len(transitions)} transitions to {file_path}")


def network(obs):
    return np.array([0., 0., 0., 0., 0., 0., 0.])


def get_upvla_agent(model_config, cfg):
    #########################
    # showo_vla prepare  #
    #########################
    device = torch.device(f"cuda:{cfg.device}")
    config = model_config
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting_w_action(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>",
                        "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob)

    def get_vq_model_class(model_type):
        if model_type == "magvitv2":
            return MAGVITv2
        else:
            raise ValueError(f"model_type {model_type} not supported.")

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Upvla.from_pretrained(
        config.model.showo.pretrained_model_path, low_cpu_mem_usage=False, act_step=config.act_step).to(device)
    assert config.model.showo.vocab_size == model.vocab_size
    # load from tuned ckpt
    path = f"{config.model.showo.tuned_model_path}/unwrapped_model/pytorch_model.bin"
    print(f"Resuming from checkpoint {path}")
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    mask_token_id = model.config.mask_token_id
    return (model_config, model, uni_prompting, vq_model, mask_token_id)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--name",
        type=str,
        default="unname",
    )
    arg_parser.add_argument(
        "--num_demos",
        type=int,
        default=5,
    )
    args = arg_parser.parse_args()

    env = gym.make("FrankaPick-Vision-v0", save_video=False)
    # env = GripperCloseEnv(env) !!!!!!!
    # env = SpacemouseIntervention(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [k for k in env.observation_space.keys() if "state" not in k]

    # from serl_launcher.networks.reward_classifier import load_classifier_func
    # import jax

    # rng = jax.random.PRNGKey(0)
    # rng, key = jax.random.split(rng)
    # classifier_func = load_classifier_func(
    #     key=key,
    #     sample=env.observation_space.sample(),
    #     image_keys=image_keys,
    #     checkpoint_path="/home/undergrad/code/serl_dev/examples/async_cable_route_drq/classifier_ckpt/",
    # )
    # env = BinaryRewardClassifierWrapper(env, classifier_func)
    ###################################################
    print("load model")
    from omegaconf import OmegaConf

    model_config = OmegaConf.load("./upvla_model.yaml")
    model = get_upvla_agent(model_config)
    print("model loaded success")

    ####################################################
    env = ZeroRewardWrapper(env)
    obs, _ = env.reset()
    traj_start = time.time()
    # agent.reset()

    transitions = []
    success_count = 0
    success_needed = args.num_demos
    total_count = 0

    pbar = tqdm(total=success_needed)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./collect_data/{uuid}_{args.name}"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")
    start = time.time()
    step = 0
    batch_size = 1
    model_config, model, uni_prompting, vq_model, mask_token_id = model
    action_buffer = None
    images_to_save_now = None
    while success_count < success_needed:

        # /home/user/serl/examples/async_pick_drq/supervised_roll_out/test_checkpoint
        ###########################################################
        image_ori_static = np.array(obs["wrist_1"][..., :3], dtype=np.uint8)[0]
        image_ori_gripper = np.array(obs["wrist_2"][..., :3], dtype=np.uint8)[0]
        # plt.savefig(f"./exp_img/{agent.step_counter}.jpg",image)
        # image = plt.
        # instruction = "pick strawberry and place in yellow bowl"
        # instruction = "pick the blue block"
        instruction = "open the drawer"
        instruction = "route the cable"
        instruction = "pick the yellow block"
        # instruction = "place the object in red plate"
        # instruction = "press the button"
        # instruction = "pick the orange arrow"
        if step % model_config.act_step == 0:
            pixel_values_static = image_transform(
                Image.fromarray(np.uint8(image_ori_static)),
                resolution=model_config.dataset.preprocessing.resolution).to(model.device).unsqueeze(0)
            image_tokens = vq_model.get_code(pixel_values_static) + len(uni_prompting.text_tokenizer)
            if model_config.model.vla.num_view == 2:
                pixel_values_gripper = image_transform(
                    Image.fromarray(np.uint8(image_ori_gripper)),
                    resolution=model_config.dataset.preprocessing.resolution).to(model.device).unsqueeze(0)
                image_tokens_gripper = vq_model.get_code(pixel_values_gripper) + len(uni_prompting.text_tokenizer)
                image_tokens = torch.cat([image_tokens, image_tokens_gripper], dim=1)
            input_ids, _ = uni_prompting(([instruction], image_tokens), 'pre_gen')
            attention_mask = create_attention_mask_predict_next_for_future_prediction(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True)

            # action = model.step(obs, goal)
            # print('obs_max:',obs["rgb_obs"]['cond_static'].max())
            # print('obs_shape:', obs["rgb_obs"]['cond_static'].shape)
            with torch.no_grad():
                gen_token_ids, actions = model.pre_pad_predict(
                    input_ids=input_ids,
                    uncond_input_ids=None,
                    attention_mask=attention_mask,
                    guidance_scale=None,
                    temperature=None,
                    timesteps=None,
                    noise_schedule=None,
                    noise_type=None,
                    predict_all_tokens=None,
                    seq_len=model_config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=model_config,
                    return_actions=True,
                )
            action_buffer = actions.squeeze().detach()

            # process images
            if model_config.model.vla.num_view == 1:
                image_tokens = [image_tokens]
                gen_token_ids = [gen_token_ids]
            elif model_config.model.vla.num_view == 2:
                image_tokens_ori_static, image_tokens_ori_gripper = image_tokens.chunk(2, dim=1)
                image_tokens = [image_tokens_ori_static, image_tokens_ori_gripper]
                gen_token_ids_static, gen_token_ids_gripper = gen_token_ids.chunk(2, dim=1)
                gen_token_ids = [gen_token_ids_static, gen_token_ids_gripper]
            else:
                raise NotImplementedError(f"Num-view {model_config.model.vla.num_view} not supported")

            images_to_save_new = []
            for i, (image_tokens_i, gen_token_ids_i) in enumerate(zip(image_tokens, gen_token_ids)):
                gen_token_ids_i = torch.clamp(gen_token_ids_i, max=model_config.model.showo.codebook_size - 1, min=0)
                gen_images = vq_model.decode_code(gen_token_ids_i)
                # Convert to PIL images
                gen_images = torch.clamp((gen_images + 1.0) / 2.0, min=0.0, max=1.0)
                gen_images *= 255.0
                gen_images = gen_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                # print(image_tokens_i- len(uni_prompting.text_tokenizer))
                recons_images = vq_model.decode_code(image_tokens_i - len(uni_prompting.text_tokenizer))
                recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
                recons_images *= 255.0
                recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                images_to_save_new.append((recons_images, gen_images))
                # save predicted images of last turn
            # print(images_to_save_now)
            if images_to_save_now is not None:
                # equals to step!=0
                images_to_save = [
                    np.concatenate([recons_images_before, gen_images_before, recons_images_new], axis=2)
                    for (recons_images_before, gen_images_before), (recons_images_new,
                                                                    _) in zip(images_to_save_now, images_to_save_new)
                ]
                images_to_save = np.concatenate(images_to_save, axis=1)
                pil_images = Image.fromarray(images_to_save.squeeze())
                os.makedirs(f"{str(model_config.model.showo.tuned_model_path)}/input_predict_truth", exist_ok=True)
                save_path = f"{str(model_config.model.showo.tuned_model_path)}/input_predict_truth/{instruction}_step_{step:03}.png"
                pil_images.save(save_path)

            images_to_save_now = images_to_save_new

        actions = action_buffer[step % model_config.act_step]
        print("action", actions, "cost", time.time() - start)
        if actions[-1] < -0.9:
            print(step)
            traj_end = time.time()
            print("traj time", traj_end - traj_start)
            start = time.time()
        ###########################################################
        next_obs, rew, done, truncated, info = env.step(action=actions)
        step += 1
        # actions = info["intervene_action"]
        # print(actions[-1])
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            ))
        transitions.append(transition)

        obs = next_obs

        if done:
            success_count += 1
            total_count += 1
            print(f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed.")
            pbar.update(1)
            obs, _ = env.reset()
            traj_start = time.time()
            # agent.reset()
            print("save trajectory")
            traj_length = env.max_episode_length
    # save_img(transitions,file_path=file_path,traj_length=traj_length)
    # transitions = []

    # with open(file_path, "wb") as f:
    #     pkl.dump(transitions, f)
    #     print(
    #         f"saved {success_needed} demos and {len(transitions)} transitions to {file_path}"
    #     )

    env.close()
    pbar.close()
