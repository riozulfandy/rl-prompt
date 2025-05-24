# RLPrompt

This repository contains the codebase for **RLPrompt**, a discrete prompt optimization framework as described in the paper:
**[RLPrompt: Optimizing Discrete Text Prompts With Reinforcement Learning](https://arxiv.org/abs/2205.12548)**
Authors: Mingkai Deng\*, Jianyu Wang\*, Cheng-Ping Hsieh\* (equal contribution), Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric P. Xing, Zhiting Hu

This implementation includes modifications to support three different RL algorithms: **PPO**, **Deep Q-Learning (DQN)**, and **Soft Q-Learning (SQL)**.

---

## Getting Started

Recent research (e.g., [this paper](https://arxiv.org/abs/2107.13586)) shows that *prompting* pre-trained language models (LMs) with specific text can guide them to perform various NLP tasks—*without* updating the model's parameters.

However, most previous works either:

* Tuned soft prompts using gradient-based optimization
* Searched for discrete prompts using heuristics

In contrast, **RLPrompt** formulates discrete prompt optimization as a **reinforcement learning problem**, training a policy network to generate prompts that maximize a reward signal.

Compared to soft prompts, **discrete prompts** are:

* Lightweight
* Interpretable
* Transferable across model architectures (e.g., RoBERTa → GPT-2) and scales (e.g., small → large models)

📖 See the full paper for more insights: [RLPrompt on arXiv](https://arxiv.org/abs/2205.12548)

![Framework Overview](figure.png)

---

## How to Run

### 1. Setup

Requirements:

* Python ≥ 3.7
* PyTorch ≥ 1.10.1 ([install here](https://pytorch.org/get-started/locally/))

Install dependencies:

```bash
pip install -e .
```

---

### 2. Download Dataset

Run the following script (for Windows):

```powershell
.\download-and-extract.ps1
```

Or download manually from [Kaggle](https://www.kaggle.com/datasets/riozulfandy04/rl-prompt-16-shot-classification-dataset), extract, and put in `examples/few-shot-classification/data/16-shot`

---

### 3. Run Training

Navigate to the training directory:

```bash
cd examples/few-shot-classification
```

Run a 16-shot classification experiment using:

```bash
python run_fsc.py \
  dataset=[sst-2, yelp-2, mr, cr, agnews, sst-5, yelp-5] \
  dataset_seed=[0, 1, 2, 3, 4] \
  prompt_length=[optional integer, default=2] \
  task_lm=[distilroberta-base, roberta-base, roberta-large, \
           distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
  random_seed=[optional integer]
```

Adjust additional hyperparameters in `fsc_config.yaml`:

```yaml
# Few-Shot Classification Config
defaults:
 - base_fsc
 - _self_

num_shots: 16
base_path: "./data"
dataset: "sst-2"
dataset_seed: 0
task_lm: "distilroberta-base"

prompt_length: 2
prompt_train_batch_size: 16
prompt_infer_batch_size: 1

reward_shaping_old_min: 0
reward_shaping_old_max: 1
reward_shaping_new_min: 0
reward_shaping_new_max: 5
top_k: 256
gamma: 0.99

max_train_steps: 6000
train_shuffle: false
eval_steps: 10
save_steps: 100
learning_rate: 5e-5
random_seed: null

training_mode: "ppo-onpolicy"  # Options: ppo-onpolicy, q-onpolicy, sql-onpolicy
```


**Module (Algorithm) Default Configuration**

```python
class ModuleConfig:
    # SQL-specific
    sql_loss_impl = "v2_v2r_v3_v3r"
    
    # SQL & QL
    target_update_method = "polyak"
    target_update_steps = None
    target_learning_rate = 0.001
    
    # PPO-specific
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    lam = 0.95
    
    # PPO & QL
    gamma = 0.99
    
    # General
    training_mode = "ppo-onpolicy"
    mix_strategy = None
    reward_shaping = True
    reward_shaping_old_min = 0
    reward_shaping_old_max = 1
    reward_shaping_new_min = 0
    reward_shaping_new_max = 5
    top_k = None
    top_p = 1.0
    num_beams = 1
```

---

## 4. Run Evaluation

After training, evaluate the learned prompt with:

```bash
cd evaluation
python run_eval.py \
  dataset=[sst-2, yelp-2, mr, cr, agnews, sst-5, yelp-5] \
  task_lm=[distilroberta-base, roberta-base, roberta-large, \
           distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
  prompt="Your Prompt Here"
```

💡 Note: For prompts with leading spaces, wrap them as `prompt=" Absolutely"`

### Example Prompts (from best experiments with `distilroberta-base`):

* **SST-2**: `Graphics equally`, `extremely extremely`, `Thus deeply`
* **SST-5**: `uatesude`, `inessiness`, `Bot animation`

---

## Experiment Details

All experimental results are available in the [`experiments`](experiments) directory. They include:

* RLPrompt results with PPO, DQN, and SQL
* Supervised fine-tuning baselines for comparison
* Training and validation performance visualizations
