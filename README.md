<div align="center">
<h2><center>👉 UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent</h2>

[Jianke Zhang*](), [Yanjiang Guo*](), [Yucheng Hu*](), [Xiaoyu Chen](), [Xiang Zhu](), [Jianyu Chen]()


<a href='https://arxiv.org/abs/2501.18867'><img src='https://img.shields.io/badge/ArXiv-2501.18867-red'></a> 
<!-- <a href='https://sites.google.com/view/pad-paper'><img src='https://img.shields.io/badge/Project-Page-Blue'></a>  -->

</div>
<div align=center>
<img src="gallery/upvla.png" alt="UP-VLA samples" align="middle"/>
</div>



This repo is the official PyTorch implementation for ICML 2025 paper [**UP-VLA**](https://arxiv.org/abs/2501.18867).

<!-- ## Friendship Link 🔥

🔥🔥🔥**Dec. 2024:** We are excited to announce our latest work [**Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations**](https://video-prediction-policy.github.io/) which is even stronger and faster. Video-Prediction-Policy finetune a video foundation model on manipulation domain with internet maniplation datasets to guide action learning. -->


##  Installation 🛠️
First, download and set up the environment.

```bash
git clone https://github.com/CladernyJorn/UP-VLA.git
pip install -r requirements.txt
```
Login your wandb account on your machine or server.
```bash
wandb login <your wandb keys>
```

If you want to perform experiments in [Calvin](https://arxiv.org/pdf/2112.03227), you need also prepare with the calvin environment following the official repo of [Calvin](https://github.com/mees/calvin.git).

Download [showlab/show-o-w-clip-vit-512x512](https://huggingface.co/showlab/show-o-w-clip-vit-512x512) and or other show-o backbone you want from the huggingface. UP-VLA is built on the backbone of `show-o-w-clip-vit-512x512` by default. Prepare the backbone checkpoints under the `./showlab` folder.

## Data Preparation
### Embodied Data
(1) Choice one:

Download [Calvin](https://github.com/mees/calvin.git) dataset and [Bridge](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0) dataset (you can skip the bridge dataset during pretraining), and process the raw data with script in `./preprocess_data`:
```bash
cd preprocess_data
# modify the path in scripts
python process_calvin.py
python process_bridge.py
```

(2) Choice two (better for using your own robot data or dataloader):
See the implementation of DataProvider class in "training/future_view_predction_w_action_dataset.py" and reimplement the dataloader class fit your dataset.

### MMU Data
We also use the [llava_tuning_665k_data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) for cotraining to maintain model's multimodal understanding capability. If you don't want to cotrain with MMU dataset for training, you can modify the config file and exclude the mmu dataloader in `train_upvla.py`.

## Train UP-VLA 🛸 
### 🛸 Training requirements
Our experiments are run on 4 A800 80G GPU. Under this setting, the training process takes ~70G GPU memory. 

If you have limited GPU memory, you can modify the batchsize setting in `config/yaml` config files. 

### 🛸 Training Pipeline
(1) Prediction and Understanding Pretraining：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml --main_process_port=8888 train_upvla.py config=./config/upvla_pred_tuning.yaml
```

(2) Prediction and Understanding Pretraining：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file ./accelerate_configs/4_gpus_deepspeed_zero2.yaml --main_process_port=8888 train_upvla.py config=./config/upvla_action_tuning.yaml
```

You can skip the pretraining stage or using MMU dataset cotraining for policy learning by modify the data_path and coeff arguments in config files.


## Evaluation 📊

### 📊 Rollout on Calvin benchmark
You should install Calvin as described in installation section. Remember to reset the `dataset_path, root_data_dir` in `policy_rollout/calvin_evaluate_upvla.yaml` with the origin calvin abcd dataset. Then, you need modify `tuned_model_path` in `policy_rollout/upvla_model.yaml` to specify the checkpoint. You can directly use our provided checkpoint or your saved checkpoint using our training script.

Lastly, you need to copy the scripts to the CALVIN evaluation folder (installed with the official repo) and run: 
```bash
cd /path to your calvin folder
python evaluation/calvin_evaluate_upvla.py
```
After running this command, you can find the predicted images in the folder of `tuned_model_path` which visualize both the current observations and future predictions.

### 📊 Rollout in your own embodiments
For your own data, you should first train the model with your own dataloader. For rollout, we provide a script `./policy_rollout/policy_upvla.py` as a reference, which can be directly used in Franka Emika Robotarm.

## CheckPoints 📷
For reproduction results on Calvin dataset, we provide trained [checkpoint](https://huggingface.co/CladernyJorn/UP-VLA-Calvin/tree/main) on Calvin ABC-D task for download.

## Bibtex 
🌟 If you find our work helpful, please leave us a star and cite our paper. Thank you!
```
@article{zhang2025up,
  title={UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent},
  author={Zhang, Jianke and Guo, Yanjiang and Hu, Yucheng and Chen, Xiaoyu and Zhu, Xiang and Chen, Jianyu},
  journal={arXiv preprint arXiv:2501.18867},
  year={2025}
}
## Acknowledgments
This work is based on [Show-o](https://github.com/showlab/Show-o), [Phi-1.5](https://huggingface.co/microsoft/phi-1_5) and [LLaVA](https://github.com/haotian-liu/LLaVA). Thanks to all the authors for their great work.