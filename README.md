# OMSD: Offline MARL with Sequential Score Decomposition


## Notification
This branch is used for testing MPE tasks and MAMujoco tasks. The MPE datasets are from OMAR 2021. The MAMujoco datasets are from MADiff 2024, which is  collected by OG-MARL with new version MAMujoco envs mujoco-210. 

Here we provide mujoco210 files in ```diffmarl/downloads/mujoco210```. Otherwise, you may also download them from the website:

```
MUJOCO_DIR=~/.mujoco
mkdir -p $MUJOCO_DIR
# 210
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O $MUJOCO_DIR/mujoco210.tar.gz
tar -xzf $MUJOCO_DIR/mujoco.tar.gz -C $MUJOCO_DIR
rm $MUJOCO_DIR/mujoco.tar.gz
```

Last, you need to set environment config in ```~/.bashrc```:
```
# mujoco210
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib64/libGLEW.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```


Test the mamujoco 210:
```
python diffmarl/test_mujoco210.py
```

Notice:
Currently we install the env with madiff as 

```bash
sudo apt-get update
sudo apt-get install libssl-dev libcurl4-openssl-dev swig
conda create -n omsd python=3.8
conda activate omsd
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt  (madiff)
# setup mamujoco
pip install -e third_party/multiagent_mujoco
pip install Cython==0.29.28
pip install tensorboard==2.11.0
pip install tensorboard-logger==0.1.0
pip install wandb==0.19.3
pip install seaborn==0.13.2
```



## Source Repo
From Diffusion-QL

- Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning<br>
Zhendong Wang, Jonathan J Hunt and Mingyuan Zhou <br>
https://arxiv.org/abs/2208.06193 <br>

and OMAR

- The multi-agent framework is based on the open-source [OMAR](https://github.com/ling-pan/OMAR) framework and CFCQL, and please refer to that repo for more documentation.

and SRPO

- Score regularization policy optimization

## Time list

- v0.1 Jan. 30, 2024. This is the version for JAL_Diffusion QL and Ind_Diffusion QL. 
- v0.2 Mar. 25, 2024. Update JAL_SRPO
- v0.3 April 15, 2024. Update IND_SRPO, CTDE(SEQ)_SRPO
- v0.4 April 30, 2024. Update new CTDE_SRPO, SEQ_SRPO, CTDE others, SEQ others. Performance bad on expert datasets.
- v0.4 May 13, 2024. Delete algo with "others". Shortern MASRPO classes with hieration.
- v0.5 Sep 30, 2024. Add mix_datasets.ipynb for mix datasets across multiple seeds. Add curve_plot.py for Ablation Study 1.
- v0.6 Jan 18, 2025. 训练dataset 3 expert+dataset 1 medium，检查数据集对训练结果的影响。更换ogmarl数据集。
- v0.7 April 1, 2025. 结合ICML审稿意见，扩展mamujoco实验，以及检查pretrain critic/diffusion的reward质量和t-SNE质量。


TODO

- 验证IQL质量，尤其是mid-replay为什么总是训练特别快还容易爆炸
- 验证diffusion效果
- 扩展更多mamujoco


## Code Structure

diffmarl/

- main.py
    - make_parallel_env
    - eval_policy
    - log_and_print
    - load_SRPO_critic
    - load_SRPO_diffusion

    - offline_train
        - make(env_id)
        - kwargs
        - ma_agent & algo_name
        - load critic and diffusion for SRPO
        - load DDPG for preys
        - replay_buffer loading
        - config.json dump
        - training process
            - prep training
            - load samples
            - update with samples

    - main()
        - args = parser
        - offline_train(args)

- algorithms
    - MASRPO
        - IND_SRPO
            - self.agents = [SRPO/SRPO_CTDE]
            - self.preys = [DDPG]
            - step
            - update
            - prep_training/rollout
            - load_pretrain_preys
            - init_from_env
        - JAL_SRPO
        - SEQ_SRPO
    - SRPO
        - SRPO(CTDE)
            - self.diff = ScoreNet_IDQL
            - self.policy = Dilac
            - self.q = IQL_Critic
            - update policy

        - SRPO_behavior
            - self.diff = ScoreNet_IDQL
            - update behavior

        - SRPO_IQL
            - self.deter_policy = Dilac
            - self.q = IQL_Critic
            - update_iql

        - IQL_Critic
            - self.q = TwinQ
            - self.vf = V
            - update q with L2 loss + soft target


- pretrain_behavior.py
    - train_IND/JAL_behavior
    - critic(args)
    - get args
    - SRPO_behavior

- pretrain_critic.py
    - train_IND/JAL_critic
    - critic(args)
    - get args
    - SRPO_IQL

- SRPO_premodels
    - expid
        - JAL
        - CTDE
        - IND
            - seed

- datasets
    - Mujoco
    - MPE

- utils
- results
- envs


## 环境修改

这里修改了原代码的 utils/env_wrappers，返回的next obs是一个list而不是array来兼容MPE和网络输入

next_obs = np.array(next_obs)

所以在使用和测试MPE时，要给next obs额外加一个array操作

## Requirements

- Multi-agent Particle Environments: in envs/multiagent-particle-envs and install it by `pip install -e .`
- python: 3.9 
- torch > 1.4 by hand (`conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge`)
- gym==0.10.8
- [MuJoCo==2.0](roboti.us/download.html): The dataset is sampled from MuJoCo2.0. Using mujoco>=2.1 will cause great performance drop 
- Install Guideline
    - DI-engine(https://di-engine-docs.readthedocs.io/zh-cn/latest/13_envs/mujoco.html?highlight=mujoco)
    - zhihu: old mujoco 200 (https://zhuanlan.zhihu.com/p/352304615)
    - 或者使用gymnasium，无需安装mujoco py和dm tree: ``` pip install gymnasium[mujoco] ```
- mujoco_py200 
- Multi-agent MuJoCo: Please check the [multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco) repo for more details about the environment. You can use the copy "multiagent_mujoco" in this directory without installation directly.
- If ```pip3 install -U 'mujoco-py<2.1,>=2.0'``` fails, try ```pip install mujoco_py==2.0.2.8``` instead to install old version mujoco-py, corresponding to mujoco200. Try ```pip install mujoco-py==2.1.2.14``` to install new version mujoco-py, corresponding to mujoco210.


Use pip to install all dependencies:
`pip install -r requirements.txt`



## Datasets
Datasets for different tasks are available at the following links uploaded by the author of [OMAR](https://github.com/ling-pan/OMAR). Please download the datasets and decompress them to the datasets folder.
- [HalfCheetah](https://drive.google.com/file/d/1zELoWUZoy3wPpwYni9t_TbzOjF4Px2f0/view?usp=sharing)
- [Cooperative Navigation](https://drive.google.com/file/d/1YVk_ajtvbcq8R2m0u0RasfB0csToV7XP/view?usp=sharing)
- [Predator-Prey](https://pan.baidu.com/s/16W-UyyCtfKDt9oTgeNOhJA): password is m7vw
- [World](https://pan.baidu.com/s/1pjZmeIAlaepPpug3b5olGA): password is 5k3t

Note: The datasets are too large, and the Baidu (Chinese) online disk requires a password for accessing it. Please just enter the password in the input box and click the blue button. The dataset can then be downloaded by cliking the "download" button (the second white button).

## Usage

Please follow the instructions below to replicate the results in the paper. 

To run independent vanilla DiffusionMARL, run the following:

```
python main.py --env_id <ENVIRONMENT_NAME> --data_type <DATA_TYPE> --dataset_num <DATASET_NUM>
```

- env_id: simple_spread/simple_tag/simple_world/HalfCheetah-v2
- data_type: random/medium-replay/medium/expert
- dataset_num: 0/1/2/3/4

For example, you can run

```
python main.py --env_id HalfCheetah-v2 --data_type expert --dataset_num 0 --device 0 --seed 1
```


