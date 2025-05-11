from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
from PIL import Image
import sys

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
import importlib
import models

importlib.reload(models)
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist

# packages from origin calvin

from evaluation.multistep_sequences import get_sequences
from evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition, join_vis_lang
from models.utils.utils import get_last_checkpoint
from models.rollout.rollout_video import RolloutVideo

logger = logging.getLogger(__name__)


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, cfg, log_dir=None):
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    sequences = get_sequences(cfg.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        # epoch = checkpoint.stem.split("=")[1]
        epoch = checkpoint
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
        wandb.log({
            "avrg_performance/avg_seq_len": avg_seq_len,
            "avrg_performance/chain_sr": chain_sr,
            "detailed_metrics/task_info": task_info
        })
        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")


def evaluate_policy(model, env, lang_embeddings, cfg, num_videos=0, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    # video stuff
    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    eval_sequences = get_sequences(cfg.num_sequences)

    results = []
    plans = defaultdict(list)

    if not cfg.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        record = i < num_videos
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, lang_embeddings,
                                   val_annotations, cfg, record, rollout_video, i)
        results.append(result)
        if record:
            rollout_video.write_to_tmp()
        if not cfg.debug:
            success_rates = count_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)
        if result < 4 and record:
            rollout_video._log_currentvideos_to_file(i, save_as_video=True)

    # if num_videos > 0:
    #    print('save_video_2:',rollout_video.save_dir)
    #    # log rollout videos
    #    rollout_video._log_videos_to_file(0, save_as_video=True)
    return results, plans


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg,
                      record, rollout_video, i):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        if record:
            rollout_video.new_subtask()
        success = rollout(env, model, task_checker, cfg, subtask, lang_embeddings, val_annotations, record,
                          rollout_video)
        if record:
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, cfg, subtask, lang_embeddings, val_annotations, record=False, rollout_video=None):
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal['lang_text'] = val_annotations[subtask][0]

    model_config, model, uni_prompting, vq_model, mask_token_id = model

    start_info = env.get_info()
    batch_size = 1
    action_buffer = None
    images_to_save_now = None
    for step in range(cfg.ep_len):
        if step % model_config.act_step == 0:
            img_static = obs["rgb_obs"]["rgb_static"].squeeze().permute(1, 2, 0).cpu().numpy()
            image_ori_static = (img_static + 1.0) / 2.0
            image_ori_static *= 255.0
            pixel_values_static = image_transform(
                Image.fromarray(np.uint8(image_ori_static)),
                resolution=model_config.dataset.preprocessing.resolution).to(model.device).unsqueeze(0)
            image_tokens = vq_model.get_code(pixel_values_static) + len(uni_prompting.text_tokenizer)
            if model_config.model.vla.num_view == 2:
                img_gripper = obs["rgb_obs"]["rgb_gripper"].squeeze().permute(1, 2, 0).cpu().numpy()
                image_ori_gripper = (img_gripper + 1.0) / 2.0
                image_ori_gripper *= 255.0
                pixel_values_gripper = image_transform(
                    Image.fromarray(np.uint8(image_ori_gripper)),
                    resolution=model_config.dataset.preprocessing.resolution).to(model.device).unsqueeze(0)
                image_tokens_gripper = vq_model.get_code(pixel_values_gripper) + len(uni_prompting.text_tokenizer)
                image_tokens = torch.cat([image_tokens, image_tokens_gripper], dim=1)
            instruction = goal['lang_text']
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
            if images_to_save_now is not None and record:
                # equals to step!=0
                images_to_save = [
                    np.concatenate([recons_images_before, gen_images_before, recons_images_new], axis=2)
                    for (recons_images_before, gen_images_before), (recons_images_new,
                                                                    _) in zip(images_to_save_now, images_to_save_new)
                ]
                images_to_save = np.concatenate(images_to_save, axis=1)
                pil_images = Image.fromarray(images_to_save.squeeze())
                os.makedirs(f"{str(rollout_video.save_dir)}/input_predict_truth", exist_ok=True)
                save_path = f"{str(rollout_video.save_dir)}/input_predict_truth/{instruction}_step_{step:03}.png"
                pil_images.save(save_path)

            images_to_save_now = images_to_save_new
        obs, _, _, current_info = env.step(action_buffer[step % model_config.act_step])
        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record:
        rollout_video.add_language_instruction(lang_annotation)
    return False


@hydra.main(config_path="./", config_name="calvin_evaluate_upvla")
def main(cfg):
    log_wandb = cfg.log_wandb
    torch.cuda.set_device(cfg.device)
    seed_everything(0, workers=True)  # type:ignore
    # evaluate a custom model
    # checkpoints = [get_last_checkpoint(Path(cfg.train_folder))]
    from omegaconf import OmegaConf
    model_config = OmegaConf.load(cfg.model_config)
    # print(model_config.experiment)
    lang_embeddings = None
    env = None
    results = {}
    plans = {}
    print(cfg.device)
    env, _, lang_embeddings = get_default_beso_and_env(
        dataset_path=cfg.dataset_path,
        env=env,
        lang_embeddings=lang_embeddings,
        device_id=cfg.device,
        cfg=cfg,
    )

    device = torch.device(f"cuda:{cfg.device}")
    checkpoint = model_config.model.showo.tuned_model_path
    model = get_upvla_agent(model_config, cfg)
    if log_wandb:
        log_dir = get_log_dir(model_config.model.showo.tuned_model_path + "/calvin_evaluation")
        os.makedirs(log_dir / "wandb", exist_ok=False)
        results[checkpoint], plans[checkpoint] = evaluate_policy(
            model, env, lang_embeddings, cfg, num_videos=cfg.num_videos, save_dir=Path(log_dir))
        # print_and_save(results, plans, cfg, log_dir=log_dir)
        # run.finish()


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
        cond_dropout_prob=config.training.cond_dropout_prob,
        future_steps=config.act_step)

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
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
