# Diffusion-QL Copyright 2022 Twitter, Inc and Zhendong Wang.
# Framework copyright. CFCQL and OMAR

# Algorithm: JAL_DQ, ind_DQ, ind_SRPO
# ToDO Algo: JAL_SRPO, CTDE_SRPO, CTDE_DQ, Pretrain diffusion & critic

import os, sys, tempfile
import json
import argparse
import numpy as np
import datetime
import random
from tqdm import tqdm

# import gym
from gym.spaces import Box, Discrete
import torch
from torch.autograd import Variable
from tensorboard_logger import log_value, configure

from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import DummyVecEnv

# env check
try:
    from multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('MujocoMulti not installed')

from algorithms.madiffQL import MADiff, MADiff_JAL  #MADiff_seq, MADiff_CTCE
from algorithms.MASRPO import IND_SRPO, JAL_SRPO    # VD_SRPO, Seq_SRPO

# make parallel MA-Env
def make_parallel_env(env_id, seed, discrete_action):

    def get_env_fn(rank):
        env = make_env(env_id, discrete_action=discrete_action)
        env.seed(seed + rank * 1000)
        np.random.seed(seed + rank * 1000)
        return env

    return DummyVecEnv([get_env_fn(0)])

# evaluate policy in eval module with envs(seed+100)
def eval_policy(agent, env_name, seed, eval_episodes, discrete_action, device, env_args=None):
    if env_name in ['HalfCheetah-v2']:
        env = MujocoMulti(env_args=env_args)
        env.seed(seed + 100)

        all_episodes_rewards = []
        for ep_i in range(eval_episodes):
            agent.prep_rollouts(device=device)  # 转成 eval 模式

            env.reset()
            done = False
            episode_reward = 0.

            while not done:
                obs = env.get_obs()
                torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
                torch_agent_actions = agent.step(torch_obs, explore=False)
                if torch.is_tensor(torch_agent_actions):
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]  # 从 tensor([[a], [a], [a]]) 变为 list[np[], np[], np[]] 
                elif isinstance(torch_agent_actions, np.ndarray):
                    agent_actions = torch_agent_actions

                actions = [ac.squeeze(0) for ac in agent_actions]  # 变为 list[np, np, np]
                
                reward, done, info = env.step(actions)  
                episode_reward += reward

            all_episodes_rewards.append(episode_reward)
        
        mean_episode_reward = np.mean(np.array(all_episodes_rewards))
        return mean_episode_reward
    else:
        avg_predator_return = 0.
    
        env = make_parallel_env(env_name, seed + 100, discrete_action)

        for ep_i in range(0, eval_episodes):
            obs = env.reset()
            agent.prep_rollouts(device=device)

            for et_i in range(config.episode_length):
                obs_len = agent.nagents
                if env_name in ['simple_tag', 'simple_world']:  # if predator-prey
                    obs_len += agent.num_preys
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(obs_len)]
                # 把 obs_dim * n_agents 的tensor变成 [n * [obs_dim*1]] 的变量

                torch_agent_actions = agent.step(torch_obs, explore=False)
                if torch.is_tensor(torch_agent_actions):
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]  # 从 tensor([[a], [a], [a]]) 变为 list[np[], np[], np[]] 
                else:
                    agent_actions = torch_agent_actions
                # agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

                actions = [agent_actions]
                next_obs, rewards, dones, infos = env.step(actions)
                
                if env_name in ['simple_tag', 'simple_world']:
                    avg_predator_return += rewards[0][0]
                else:
                    avg_agent_reward = np.mean(rewards[0])
                    avg_predator_return += avg_agent_reward

                obs = next_obs

        avg_predator_return /= eval_episodes
        return avg_predator_return

# log params to tensorboard
def log_and_print(key, value, t, multi=False):
    if multi:
        print("t:", t, end=" | ")
        for i in range(len(key)):
            end = " | " if i < len(key) - 1 else "\n"
            print("{}: {:.3f}".format(key[i], value[i][0]), end=end)
            log_value(key[i], value[i], t)    
    else:
        print("t:{}, {}: {:.3f}".format(t, key, value))
        log_value(key, value, t)


def offline_train(config):
    unique_token = "{}__{}__seed{}".format(config.data_type, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"), config.seed)

    unique_token = config.marltype+"_"+unique_token
    # JAL/IND/VD/SEQ _ time _ seed 

    if not config.no_log:
        outdir = os.path.join(config.dir, config.env_id, unique_token)
        os.makedirs(outdir)
        print('\033[1;32mOutput files are saved in {} \033[1;0m'.format(outdir))
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        env = make_parallel_env(config.env_id, config.seed, config.discrete_action)
        env_args, env_info = None, None
    else:
        env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
        env = MujocoMulti(env_args=env_args)
        env.seed(config.seed)
        env_info = env.get_env_info()
  
    kwargs={'logging_interval': config.logging_interval,
            'no_log': config.no_log,
            'train_num_steps': config.num_steps}


    # select algorithms from [JAL, ind, seq, VD] + [DiffusionQL, SRPO]
    if config.difftype == "DQL":
        if config.marltype == "JAL":
            algo_name = "JAL_Diffusion-QL"
            ma_agent = MADiff_JAL.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                                agent_alg="diffusion", adversary_alg="ddpg",
                                                gamma=config.gamma, tau=config.tau, lr=config.lr,
                                                hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                                batch_size=config.batch_size, device = config.device, **kwargs)
            print("JAL-Diff-QL agents have been created")
        elif config.marltype == "IND":
            algo_name = "IND_Diffusion-QL"
            ma_agent = MADiff.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                            agent_alg="diffusion", adversary_alg="ddpg",
                                            gamma=config.gamma, tau=config.tau, lr=config.lr,
                                            hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                            batch_size=config.batch_size, device = config.device, **kwargs)
            print("Parallel update independent Diff-QL agents have been created")
        else:
            algo_name = "SEQ/VD_Diffusion-QL"
            # ma_agent = MADiff_seq.init_from_env()   # to be finished in algo/madiffQL
            print("Sequential update and VDN Diffusion-QL agents haven't been established")
    elif config.difftype == "SRPO":
        if config.marltype == "JAL":
            algo_name = "JAL_SRPO"
            ma_agent = JAL_SRPO.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                                agent_alg="diffusion", adversary_alg="ddpg",
                                                gamma=config.gamma, tau=config.tau, lr=config.lr,
                                                hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                                batch_size=config.batch_size, device = config.device, config=config, **kwargs)
            print("JAL-SRPO agents have been created")
        elif config.marltype == "IND":
            algo_name = "IND_SRPO"
            ma_agent = IND_SRPO.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                            agent_alg="diffusion", adversary_alg="ddpg",
                                            gamma=config.gamma, tau=config.tau, lr=config.lr,
                                            hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                            batch_size=config.batch_size, device = config.device, config=config, **kwargs)
            print("Parallel update independent SRPO agents have been created")
            print(ma_agent.init_dict)
        else:
            algo_name = "SEQ/VD_SRPO"
            # ma_agent = MADiff_seq.init_from_env()   # to be finished in algo/madiffQL
            print("Sequential update and VDN Diffusion-QL agents haven't been established")
    else:
        print("Neither SRPO nor Diffusion-QL have been selected. Choose valid diffusion model")
            

    # load pretrained preys model to DDPG
    if config.env_id in ['simple_tag', 'simple_world']:
        pretrained_model_dir = './datasets/{}/pretrained_adv_model.pt'.format(config.env_id)
        ma_agent.load_pretrained_preys(pretrained_model_dir)


    # load full offline datasets to buffer
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [obsp.shape[0] for obsp in env.observation_space],
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space], device = config.device
        )
    else:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space], device = config.device
        )
    replay_buffer.load_batch_data(config.dataset_dir, rew_scale = config.rew_scale)

    if np.isinf(replay_buffer.ave_reward):   # 如果变量是 inf, 代表这条轨迹没有 done=True，要进行 scale
        replay_buffer.ave_reward = replay_buffer.sum_reward / (replay_buffer.filled_i/config.episode_length)
    print('Average_reward:', replay_buffer.ave_reward)

    # tensorboard log dir and save configurations
    if not config.no_log:
        configure(outdir)
        config_log_dict = {"env": config.env_id,
                           "dataset": "{}_{}".format(config.data_type, config.dataset_num),
                           "dataset_ave_reward": replay_buffer.ave_reward,
                           "algo": algo_name,
                           "seed": config.seed,
                           "Denoise_steps": config.T,
                           "batch size": config.batch_size,
                           "discount factor": config.gamma,
                           "soft update": config.tau,
                           }
        param_dict = os.path.join(outdir, 'config.json')
        with open(param_dict, 'w') as f:
            json.dump(config_log_dict, f)


    # training process
    ma_agent.prep_training(device=config.device)

    progress_bar = tqdm(range(config.num_steps+1), desc = 'Training Process', leave=True)

    for t in range(config.num_steps + 1):
        # set as eval() when eval
        if t % config.eval_interval == 0 or t == config.num_steps:
            print('Start to {} times eval | Timestep:{}'.format(t % config.eval_interval, t))
            eval_return = eval_policy(ma_agent, config.env_id, config.seed, config.eval_episodes, config.discrete_action, config.device, env_args=env_args)
            if not config.no_log:
                log_and_print('eval_return', eval_return, t)
                log_and_print('normed_eval_return', eval_return/replay_buffer.ave_reward, t)
            # when eval finished, switch to train()
            ma_agent.prep_training(device=config.device)
                
        # load joint datasets for JAL and indpendent trajectory for ind/seq training
        if config.marltype == "JAL":
            sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
            ma_agent.update(sample, t)  
            # 这里一个可能的问题是，JAL需不需要区分pray的数据

        elif config.marltype == "IND":
            nagents = ma_agent.nagents if config.env_id in ['simple_spread', 'HalfCheetah-v2'] else ma_agent.num_predators
            samples = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
            
            # keep each agent data in one batch
            # independent learning 只拿agent i自己的buffer，并只更新a i策略
            for a_i in range(nagents):
                sample_i = samples[a_i]
                ma_agent.update(sample_i, a_i, t)
        else:
            pass
            # 对于seq和ctde需要添加额外的更新方法
                
        progress_bar.update(1)

    try:
        env.close()
    except Exception as e:
        print(f"An error occurred while closing the environment: {e}")

# Pretrain SRPO_behavior and SRPO_critic for MARL version
        
temperature_coefficients = {"simple_spread": 0.08,
                            "halfcheetah-medium-expert-v2": 0.01,
                            "halfcheetah-medium-v2": 0.2, 
                            "halfcheetah-medium-replay-v2": 0.2}
# change into MARL version

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='simple_spread', type=str, help="Name of environment")
    parser.add_argument("--data_type", default='expert', type=str)
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")
    
    # Algo choice: Diffusion QL or SRPO
    parser.add_argument("--difftype", default='DQL') # DQL for Diffusion-QL, SRPO for SRPO algo
    # JAL for joint action learning CTCE, IND for independent learning, VD for QMIX decomposition, SEQ for sequential update/regularization
    parser.add_argument("--marltype", default='JAL') # JAL, IND, VD, SEQ

    # Set diffusion params
    parser.add_argument("--T", default=5, type=int, help="Denoising steps for DDPM")
    parser.add_argument("--beta_schedule", default='vp', type=str)
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--use_gpu", default=True, type=bool, help='use cuda or not')
    parser.add_argument("--device", default=0, type=int, help='cuda number')

    """   Unchangeable Params   """
    # log and save dir
    parser.add_argument("--dir", type=str, default='results', help="Name of directory to store model/training contents")
    parser.add_argument('--dataset_dir', default='/home/qiaodan/Code/diffmarl/datasets', type=str)

    # params for envs
    parser.add_argument("--discrete_action", action='store_true', default=False)
    
    # params for buffer and data
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--episode_length", default=25, type=int, help='MPE epi_length is 25, MAMuJoCo epi_length is 1000')
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for model training")
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--set_lr", action='store_true')
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--rew_scale", default=1.0, type=float)
    
    # params for RL
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--tau", default=0.01, type=float)

    # params for evaluation
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--eval_interval', default=10000, type=int)
    parser.add_argument('--num_steps', default=int(1e7), type=int)

    # params for logging
    parser.add_argument("--logging_interval", default=500, type=int)
    parser.add_argument("--no_log", action='store_true')

    ######### args for SRPO To be revise #########
    
    
    parser.add_argument("--save_model", default=1, type=int)       
    # parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--beta', type=float, default=None)       
    parser.add_argument('--actor_load_path', type=str, default=None)
    parser.add_argument('--critic_load_path', type=str, default=None)
    # parser.add_argument('--policy_batchsize', type=int, default=256)              
    parser.add_argument('--actor_blocks', type=int, default=3)     
    # parser.add_argument('--z_noise', type=int, default=1)
    parser.add_argument('--WT', type=str, default="VDS")
    parser.add_argument('--q_layer', type=int, default=2)
    parser.add_argument('--n_policy_epochs', type=int, default=100)
    parser.add_argument('--policy_layer', type=int, default=None)
    parser.add_argument('--critic_load_epochs', type=int, default=150)
    parser.add_argument('--regq', type=int, default=0)
    ##################################################

    config = parser.parse_args()

    # config.env 替换为 config.env_id
    if config.beta is None:
        config.beta = temperature_coefficients[config.env_id]

    if config.policy_layer is None:
        config.policy_layer=4 if "maze" in config.env_id else 2

    if "maze" in config.env_id:
        config.regq = 1

    if config.use_gpu:
        config.device = f"cuda:{config.device}"
    else:
        config.device = "cpu"

    # make envs params
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        config.lr=0.005
        config.num_steps = 25000
        if config.env_id == 'simple_world':
            config.steps_per_update=20

    else:
        config.num_steps = int(1e6)
        config.steps_per_update = 10
        config.eval_interval = 5000
        config.logging_interval = 5000
        config.episode_length=1000
        config.gamma=0.99
        config.lr = 0.0003
           
        config.tau = 0.005
        config.gamma = 0.99

    if config.marltype == 'JAL':
        config.T=20
        
    config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)
        
    offline_train(config)

