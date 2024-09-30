# This is the file to check the pretrained models of IND & JAL
# 1. the difference between JAL partial score and IND score
# 2. bandit dataset, print learned gradients and distributions, visualize the KL constraint
# 3. eval the critic, IND works, CTDE works, JAL doesn't work. Maybe JAL score is wrong???

"""用来绘制bandit的梯度场"""


# Diffusion-QL Copyright 2022 Twitter, Inc and Zhendong Wang.
# Framework copyright. CFCQL and OMAR

# Algorithm: JAL_DQ, ind_DQ, ind_SRPO, JAL_SRPO, CTDE_SRPO
# ToDO Algo: QMIX_SRPO

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

from bandit import ContinuousBanditEnv

from algorithms.madiffQL import MADiff, MADiff_JAL  #MADiff_seq, MADiff_CTCE
from algorithms.MASRPO import IND_SRPO, JAL_SRPO, CTDE_SRPO, SEQ_SRPO     # VD_SRPO, CTDE_SRPO

import matplotlib.pyplot as plt

import wandb


# make parallel MA-Env
def make_parallel_env(env_id, seed, discrete_action):
    def get_env_fn(rank):
        env = make_env(env_id, discrete_action=discrete_action)
        env.seed(seed + rank * 1000)
        np.random.seed(seed + rank * 1000)
        return env
    return DummyVecEnv([get_env_fn(0)])

# evaluate policy in eval module with envs(seed+100) on cpu
def eval_policy(agent, env_name, seed, eval_episodes, discrete_action, device='cpu', env_args=None):
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
                # torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
                torch_obs = [torch.Tensor(obs[i]).unsqueeze(0).to(device)  for i in range(agent.nagents)] 
                torch_agent_actions = agent.step(torch_obs, explore=False)
                # if torch.is_tensor(torch_agent_actions):
                if all(isinstance(item, torch.Tensor) for item in torch_agent_actions):
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]  # 从 tensor([[a], [a], [a]]) 变为 list[np[], np[], np[]] 
                elif all(isinstance(item, np.ndarray) for item in torch_agent_actions):
                    agent_actions = torch_agent_actions
                actions = [ac.squeeze(0) for ac in agent_actions]  # 变为 list[np, np, np]
                reward, done, info = env.step(actions)  
                episode_reward += reward
            all_episodes_rewards.append(episode_reward)        
        mean_episode_reward = np.mean(np.array(all_episodes_rewards))
        return mean_episode_reward
    elif env_name == 'bandit':
        env = ContinuousBanditEnv()
        
        agent.prep_rollouts(device=device)  # 转成 eval 模式
        obs = env.reset()
        done = False
        torch_obs = [torch.Tensor(obs).unsqueeze(0).to(device)  for i in range(agent.nagents)]
        torch_agent_actions = agent.step(torch_obs, explore=False)
        actions = [ac.squeeze(0) for ac in torch_agent_actions]  # 变为 list[np, np, np]
        _, reward, done, info = env.step(actions)  # 可以简单点，直接把action相乘

        mean_episode_reward = np.array(reward)
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
    

# evaluate policy in eval module with envs(seed+100) on cpu
def eval_init_dilac_policy(agent, env_name, seed, eval_episodes, discrete_action, init_policy, device='cpu', env_args=None):
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
                # torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
                torch_obs = [torch.Tensor(obs[i]).unsqueeze(0).to(device)  for i in range(agent.nagents)] 
                torch_agent_actions = agent.step(torch_obs, explore=False)
                # if torch.is_tensor(torch_agent_actions):
                if all(isinstance(item, torch.Tensor) for item in torch_agent_actions):
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]  # 从 tensor([[a], [a], [a]]) 变为 list[np[], np[], np[]] 
                elif all(isinstance(item, np.ndarray) for item in torch_agent_actions):
                    agent_actions = torch_agent_actions
                actions = [ac.squeeze(0) for ac in agent_actions]  # 变为 list[np, np, np]
                reward, done, info = env.step(actions)  
                episode_reward += reward
            all_episodes_rewards.append(episode_reward)        
        mean_episode_reward = np.mean(np.array(all_episodes_rewards))
        return mean_episode_reward
    elif env_name == 'bandit':
        env = ContinuousBanditEnv()
        
        agent.prep_rollouts(device=device)  # 转成 eval 模式
        obs = env.reset()
        done = False
        torch_obs = [torch.Tensor(obs).unsqueeze(0).to(device)  for i in range(agent.nagents)]

        # print initial policy params of 5*256*256*1
        init_agent_actions = agent.step(torch_obs, explore=False)
        for id, srpo_agent in enumerate(agent.agents):
            dilac = srpo_agent.SRPO_policy
            dilac.net[-2].bias.data[0] = init_policy[id]

        torch_agent_actions = agent.step(torch_obs, explore=False)
        actions = [ac.squeeze(0) for ac in torch_agent_actions]  # 变为 list[np, np, np]



        _, reward, done, info = env.step(actions)  # 可以简单点，直接把action相乘

        mean_episode_reward = np.array(reward)
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


def load_SRPO_critic(srpo_model, load_path, srpo_type):
    if load_path is not None:
        print("loading critic...")
        if srpo_type == 'IND':
            for agent_index, srpo_i in enumerate(srpo_model.agents):
                # SRPO_premodels/exp_seed/IND/best_critic_i
                load_path_i = os.path.join(load_path, 'IND', f'best_critic_{agent_index}.pth')
                ckpt = torch.load(load_path_i, map_location=srpo_model.device)
                srpo_i.q[0].load_state_dict(ckpt)
        elif srpo_type == 'JAL':
            # SRPO_models/exp_seed/JAL/best_critic.pth
            load_path = os.path.join(load_path, 'JAL', f'best_critic.pth')
            ckpt = torch.load(load_path, map_location=srpo_model.device)
            srpo_model.agents[0].q[0].load_state_dict(ckpt)
        elif srpo_type == 'CTDE' or srpo_type == 'SEQ':
            for srpo_i in srpo_model.agents:
                # SRPO_premodels/exp_seed/JAL/best_critic  每个agent都有一个central Q，共享
                load_path_i = os.path.join(load_path, 'JAL', f'best_critic.pth')
                ckpt = torch.load(load_path_i, map_location=srpo_model.device)
                srpo_i.q[0].load_state_dict(ckpt)

    else:
        assert False

def load_SRPO_diffusion(srpo_model, load_path, srpo_type):
    # IND & CTDE load ind diffusion score, JAL load joint diffusion score
    if load_path is not None:
        print("loading actor...")
        if srpo_type == 'IND' or srpo_type == 'CTDE':
            for agent_index, srpo_i in enumerate(srpo_model.agents):
                # SRPO_premodels/exp_seed/IND/best_diffusion_i.pth
                load_path_i = os.path.join(load_path, 'IND', f'best_diffusion_{agent_index}.pth')
                ckpt = torch.load(load_path_i, map_location=srpo_model.device)
                # for k,v in ckpt.items():
                #     print("{} ckpt: {}, srpo: {}".format(k, ckpt[k].shape, srpo_i.state_dict()[k].shape))
                srpo_i.load_state_dict({k:v for k,v in ckpt.items() if "diffusion_behavior" in k}, strict=False)
        elif srpo_type == 'SEQ':
            for agent_index, srpo_i in enumerate(srpo_model.agents):
                # SRPO_premodels/exp_seed/Seq/best_diffusion_i.pth
                load_path_i = os.path.join(load_path, 'Seq', f'best_diffusion_{agent_index}.pth')
                ckpt = torch.load(load_path_i, map_location=srpo_model.device)
                srpo_i.load_state_dict({k:v for k,v in ckpt.items() if "diffusion_behavior" in k}, strict=False)
        elif srpo_type == 'JAL':
            # SRPO_premodels/exp_seed/ + JAL/diffusion.pth
            srpo = srpo_model.agents[0]
            load_path = os.path.join(load_path, 'JAL', f'best_diffusion.pth')
            ckpt = torch.load(load_path, map_location=srpo_model.device)
            srpo.diffusion_behavior.load_state_dict({k:v for k,v in ckpt.items() if "diffusion_behavior" in k}, strict=False)
    else:
        assert False


def save_a_traj(action, a0_traj, a1_traj, marltype):
    if marltype == 'JAL':
        a0 = action[:, 0].mean().cpu()
        a1 = action[:, 1].mean().cpu()
        a0_d = a0.detach().numpy()
        a1_d = a1.detach().numpy()
        a0_traj.append(a0_d)
        a1_traj.append(a1_d)
    else:
        a0 = action[0].mean().cpu()
        a1 = action[1].mean().cpu()
        a0_d = a0.detach().numpy()
        a1_d = a1.detach().numpy()
        a0_traj.append(a0_d)
        a1_traj.append(a1_d)

# 可视化所有score和q gradient
def vis_pretrain_score(marltype, ma_agent, replay_buffer, gpu_use):
    # 定义空间范围
    x = np.linspace(-1, 1, 11)
    y = np.linspace(-1, 1, 11)
    X, Y = np.meshgrid(x, y)
 
    s0 = replay_buffer.sample(x.shape[0], to_gpu=gpu_use)[0]['obs']  # TBD
    s1 = replay_buffer.sample(x.shape[0], to_gpu=gpu_use)[0]['obs']  # TBD

    for agent in ma_agent.agents:
        agent.diffusion_behavior.eval()

    fig, ax = plt.subplots()
    ax.set_title('Pretrain Models')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])

    ma_agent.agents[0] = ma_agent.agents[0].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    ma_agent.agents[1] = ma_agent.agents[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    if marltype == 'IND':
        # ind Q and ind score
        a0 = torch.from_numpy(x).to(s0.device)   # 11,1
        a1 = torch.from_numpy(y).to(s1.device)
        t = torch.zeros(a0.shape[0], device=s0.device) * 0.96 + 0.02  # t约等于0
        # random noising time t
        alpha_t, std = ma_agent.agents[0].marginal_prob_std(t)
        z = torch.randn_like(a0)

        perturbed_a0 = a0 * alpha_t[..., None] + z * std[..., None]   
        perturbed_a1 = a1 * alpha_t[..., None] + z * std[..., None]   
        # add noise to policy action, generate a_t

        with torch.no_grad():
            episilon0 = ma_agent.agents[0].diffusion_behavior(perturbed_a0, t, s0).detach()  # diffusion model prediction\
            episilon1 = ma_agent.agents[1].diffusion_behavior(perturbed_a1, t, s1).detach()  # diffusion model prediction

        detach_a0 = a0.detach().requires_grad_(True)  # dilac policy detach
        detach_a1 = a1.detach().requires_grad_(True)  # dilac policy detach
        qs0 = ma_agent.agents[0].q[0].q0_target.both(detach_a0 , s0)  # Q1(s, a) 
        q0 = (qs0[0].squeeze() + qs0[1].squeeze()) / 2.0
        qs1 = ma_agent.agents[1].q[0].q0_target.both(detach_a1 , s1)  # Q2(s, a) 
        q1 = (qs1[0].squeeze() + qs1[1].squeeze()) / 2.0


        guidance0 =  torch.autograd.grad(torch.sum(q0), detach_a0)[0].detach()
        guidance1 =  torch.autograd.grad(torch.sum(q1), detach_a1)[0].detach()

        ax.quiver(X, Y, episilon0, episilon1, color='red', label='IND Score')
        ax.quiver(X, Y, guidance0, guidance1, color='blue', label='IND Q Gradients')


    # elif marltype == 'CTDE':
    #     # JAL Q and ind score

    #     a0 = x   # 11,1
    #     a1 = y
    #     t = torch.zeros(a0.shape[0], device=s.device) * 0.96 + 0.02  # t约等于0
    #     # random noising time t
    #     alpha_t, std = ma_agent[0].marginal_prob_std(t)
    #     z = torch.randn_like(a0)

    #     perturbed_a0 = a0 * alpha_t[..., None] + z * std[..., None]   
    #     perturbed_a1 = a1 * alpha_t[..., None] + z * std[..., None]   
    #     # add noise to policy action, generate a_t

    #     with torch.no_grad():
    #         episilon0 = ma_agent[0].diffusion_behavior(perturbed_a0, t, s).detach()  # diffusion model prediction\
    #         episilon1 = ma_agent[1].diffusion_behavior(perturbed_a1, t, s).detach()  # diffusion model prediction

    #     detach_a0 = a0.detach().requires_grad_(True)  # dilac policy detach
    #     detach_a1 = a1.detach().requires_grad_(True)  # dilac policy detach
    #     qs0 = ma_agent[0].q[0].q0_target.both(detach_a0 , s)  # Q1(s, a) 
    #     q0 = (qs0[0].squeeze() + qs0[1].squeeze()) / 2.0
    #     qs1 = ma_agent[1].q[0].q0_target.both(detach_a1 , s)  # Q2(s, a) 
    #     q1 = (qs1[0].squeeze() + qs1[1].squeeze()) / 2.0


    #     guidance0 =  torch.autograd.grad(torch.sum(q0), detach_a0)[0].detach()
    #     guidance1 =  torch.autograd.grad(torch.sum(q1), detach_a1)[0].detach()

    #     ax.quiver(X, Y, episilon0, episilon1, color='red', label='IND Score')
    #     ax.quiver(X, Y, guidance0, guidance1, color='blue', label='IND Q Gradients')


    # elif marltype == 'JAL':
    #     # JAL Q and JAL score
    #     # ind Q and ind score
    #     a0 = x   # 11,1
    #     a1 = y
    #     t = torch.zeros(a0.shape[0], device=s.device) * 0.96 + 0.02  # t约等于0
    #     # random noising time t
    #     alpha_t, std = ma_agent[0].marginal_prob_std(t)
    #     z = torch.randn_like(a0)

    #     perturbed_a0 = a0 * alpha_t[..., None] + z * std[..., None]   
    #     perturbed_a1 = a1 * alpha_t[..., None] + z * std[..., None]   
    #     # add noise to policy action, generate a_t

    #     with torch.no_grad():
    #         episilon0 = ma_agent[0].diffusion_behavior(perturbed_a0, t, s).detach()  # diffusion model prediction\
    #         episilon1 = ma_agent[1].diffusion_behavior(perturbed_a1, t, s).detach()  # diffusion model prediction

    #     detach_a0 = a0.detach().requires_grad_(True)  # dilac policy detach
    #     detach_a1 = a1.detach().requires_grad_(True)  # dilac policy detach
    #     qs0 = ma_agent[0].q[0].q0_target.both(detach_a0 , s)  # Q1(s, a) 
    #     q0 = (qs0[0].squeeze() + qs0[1].squeeze()) / 2.0
    #     qs1 = ma_agent[1].q[0].q0_target.both(detach_a1 , s)  # Q2(s, a) 
    #     q1 = (qs1[0].squeeze() + qs1[1].squeeze()) / 2.0


    #     guidance0 =  torch.autograd.grad(torch.sum(q0), detach_a0)[0].detach()
    #     guidance1 =  torch.autograd.grad(torch.sum(q1), detach_a1)[0].detach()

    #     ax.quiver(X, Y, episilon0, episilon1, color='red', label='IND Score')
    #     ax.quiver(X, Y, guidance0, guidance1, color='blue', label='IND Q Gradients')

    ax.legend(loc='upper left')


    # 显示图形
    dir = '/home/qiaodan/Code/diffmarl/eval_result'
    plt.savefig(os.path.join(dir, '{}_pretrain_models.png'.format(marltype)), dpi=300, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()



# 可视化IND与JAL在bandit上的policy轨迹
# loss = epi score - beta * Q gradients
def vis_train_grad(epi, q_grad, action, figure, beta, marltype, t):
    if marltype == 'JAL':
        epi0 = epi[:, 0].mean().cpu()
        epi1 = epi[:, 0].mean().cpu()
        q_grad0 = q_grad[:, 0].mean().cpu()
        q_grad1 = q_grad[:, 0].mean().cpu()
        epi0_d = epi0.detach().numpy()
        epi1_d = epi1.detach().numpy()
        q_grad0_d = beta*q_grad0.detach().numpy()
        q_grad1_d = beta*q_grad1.detach().numpy()
        a0 = action[:, 0].mean().cpu()
        a1 = action[:, 1].mean().cpu()
        a0_d = a0.detach().numpy()
        a1_d = a1.detach().numpy()
    else:
        epi0 = epi[0].mean().cpu()
        epi1 = epi[1].mean().cpu()
        q_grad0 = q_grad[0].mean().cpu()
        q_grad1 = q_grad[1].mean().cpu()
        epi0_d = beta*epi0.detach().numpy()
        epi1_d = beta*epi1.detach().numpy()
        q_grad0_d = q_grad0.detach().numpy()
        q_grad1_d = q_grad1.detach().numpy()
        a0 = action[0].mean().cpu()
        a1 = action[1].mean().cpu()
        a0_d = a0.detach().numpy()
        a1_d = a1.detach().numpy()

    # 绘制箭头图
    # plt.figure(figsize=(8, 8))
    figure.quiver(a0_d, a1_d, epi0_d, epi1_d, color='red', label='Score Time {}'.format(t))
    figure.quiver(a0_d, a1_d, q_grad0_d, q_grad1_d, color='blue', label='Q Gradients Time {}'.format(t))
    # figure.scatter(a0_d, a1_d, c='black', label='Actions', s=3)


    




    



# # visualize distribution
# def vis_dist():


def offline_eval(config):
    unique_token = "{}__{}__{}__seed{}".format(config.env_id, config.data_type, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"), config.seed)

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
    elif config.env_id == "bandit":
        env = ContinuousBanditEnv()
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
        # JAL, IND, CTDE, SEQ, QMIX
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
        elif config.marltype == 'CTDE':
            algo_name = "CTDE_SRPO"
            ma_agent = CTDE_SRPO.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                            agent_alg="diffusion", adversary_alg="ddpg",
                                            gamma=config.gamma, tau=config.tau, lr=config.lr,
                                            hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                            batch_size=config.batch_size, device = config.device, config=config, **kwargs)
            print("Naive CTDE SRPO agents have been established")
        elif config.marltype == 'SEQ':
            algo_name = "SEQ_SRPO"
            ma_agent = SEQ_SRPO.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                            agent_alg="diffusion", adversary_alg="ddpg",
                                            gamma=config.gamma, tau=config.tau, lr=config.lr,
                                            hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                            batch_size=config.batch_size, device = config.device, config=config, **kwargs)
            print("Sequential update SRPO agents have been established")
    else:
        print("Neither SRPO nor Diffusion-QL have been selected. Choose valid diffusion model")
             
    # load pretrained critics and diffusion to SRPO
    if config.critic_load_path is not None:
        load_SRPO_critic(srpo_model=ma_agent, load_path=config.critic_load_path, srpo_type=config.marltype)
        print('Critic models are loaded to SRPO')
    else:
        print('Critic models are not loaded to SRPO')

    if config.diffusion_load_path is not None:
        load_SRPO_diffusion(srpo_model=ma_agent, load_path=config.diffusion_load_path, srpo_type=config.marltype)
        print('Diffusion models are loaded to SRPO')
    else:
        print('Diffusion models are not loaded to SRPO')

    # load pretrained preys model to DDPG
    if config.env_id in ['simple_tag', 'simple_world']:
        pretrained_model_dir = './datasets/{}/pretrained_adv_model.pt'.format(config.env_id)
        ma_agent.load_pretrained_preys(pretrained_model_dir)
    else:
        print('Prey DDPG are not loaded')

    # load full offline datasets to buffer
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world', 'bandit']:
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

    if np.isinf(replay_buffer.ave_reward):   # 如果变量是 inf, 代表这条轨迹没有 done=True，要进行 scale, 主要用于MPE
        replay_buffer.ave_reward = replay_buffer.sum_reward / (replay_buffer.filled_i/config.episode_length)
    print('Average_reward:', replay_buffer.ave_reward)

    # visualize pretrain models
    # vis_pretrain_score(config.marltype, ma_agent, replay_buffer, config.use_gpu)

    # tensorboard log dir and save configurations
    if not config.no_log:
        configure(outdir)
        config_log_dict = {"env": config.env_id,
                           "dataset": "{}_{}".format(config.data_type, config.dataset_num),
                           "dataset_ave_reward": replay_buffer.ave_reward,
                           "algo": algo_name,
                           "seed": config.seed,
                           "reg beta": config.beta,
                           "Denoise_steps": config.T,
                           "batch size": config.batch_size,
                           "discount factor": config.gamma,
                           "soft update": config.tau,
                           "IDQN hidden MLP": config.resnet_hidden_dim,
                           "IDQN state-action embed": config.resnet_hidden_dim,
                           "IDQN actor block": config.actor_blocks,
                           }
        print(config_log_dict)
        param_dict = os.path.join(outdir, 'config.json')
        with open(param_dict, 'w') as f:
            json.dump(config_log_dict, f)

    
    run = wandb.init(
    # set the wandb project where this run will be logged
    project="MASRPO_bandit_eval",
    # track hyperparameters and run metadata
    config=config_log_dict)


    # training process
    ma_agent.prep_training(device=config.device)

    progress_bar = tqdm(range(config.num_steps+1), desc = 'Training Process', leave=True)

    a0_traj = []
    a1_traj = []
    fig, ax = plt.subplots(figsize=(8, 8))
    # 添加标题和轴标签
    ax.set_title('Quiver Gradient and Score Plot')
    ax.set_xlabel('action 1')
    ax.set_ylabel('action 2')
    

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])

    




    for t in range(config.num_steps + 1):
        
        # set init dilac policy
        if t == 0:
            # eval_policy will set rollouts at start
            print('Init params')
            init_eval_return = eval_init_dilac_policy(ma_agent, config.env_id, config.seed, config.eval_episodes, config.discrete_action, init_policy=config.init_actions, device='cpu', env_args=env_args)
            ma_agent.prep_training(device=config.device)
        


        # set as eval() when eval
        if t % config.eval_interval == 0 or t == config.num_steps:
            # eval_policy will set rollouts at start
            print('Start to {} times eval | Timestep:{}'.format(t % config.eval_interval, t))
            eval_return = eval_policy(ma_agent, config.env_id, config.seed, config.eval_episodes, config.discrete_action, device='cpu', env_args=env_args)
            if not config.no_log:
                log_and_print('eval_return', eval_return, t)
                log_and_print('normed_eval_return', eval_return/replay_buffer.ave_reward, t)
            # when eval finished, switch to train()
            ma_agent.prep_training(device=config.device)
                
        # load joint datasets for JAL and indpendent trajectory for ind/seq training
        if config.marltype == "JAL":
            sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
            epi, q_grad, a_plt = ma_agent.update(sample, t, run)  
            # 这里一个可能的问题是，JAL需不需要区分pray的数据
        elif config.marltype == "IND":
            nagents = ma_agent.nagents if config.env_id in ['simple_spread', 'HalfCheetah-v2', 'bandit'] else ma_agent.num_predators
            samples = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)

            if config.env_id == 'bandit':
                epi = []
                q_grad = []
                a_plt = []
                
            # 只拿agent i自己的buffer，并只更新a i策略
            for a_i in range(nagents):
                sample_i = samples[a_i]
                epi_i, q_grad_i, a_plt_i = ma_agent.update(sample_i, a_i, t, run)
                epi.append(epi_i)
                q_grad.append(q_grad_i)  
                a_plt.append(a_plt_i)    

        elif config.marltype == "CTDE" or config.marltype == "SEQ":
            samples = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
            # 只拿agent i自己的buffer，并只更新a i策略; 但是计算Q值用的是total state，以及other policy actions
            # 一起输入给进去再分开，更新体现在ma agent内部
            epi, q_grad, a_plt = ma_agent.update(samples, t, run)
        else:  # QMIX_SRPO
            pass
        
        save_a_traj(a_plt, a0_traj, a1_traj, config.marltype)

        if t % config.print_interval == 0 or t == config.num_steps:
            # visualize gradient
            print('Print Gradients Fields | Timestep:{}'.format(t))
            vis_train_grad(epi, q_grad, a_plt, ax, config.beta, config.marltype, t)
                
        progress_bar.update(1)

    
    ax.plot(a0_traj, a1_traj,  linestyle='--', c='black', label='Actions')
    ax.legend(['Score', 'Q Gradients'], loc='upper left')

    dir = '/home/qiaodan/Code/diffmarl/eval_result'
    plt.savefig(os.path.join(dir, '{}_beta{}_seed{}_traj.png'.format(config.marltype, config.beta, config.seed)), dpi=300, format='png', bbox_inches='tight', pad_inches=0.1)

    try:
        env.close()
    except Exception as e:
        print(f"An error occurred while closing the environment: {e}")

# Pretrain SRPO_behavior and SRPO_critic for MARL version
        
temperature_coefficients = {"simple_spread": 0.08,
                            "HalfCheetah-v2": 0.02,
                            "bandit": 0.1}
# change into MARL version

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='bandit', type=str, help="Name of environment")   # HalfCheetah-v2
    parser.add_argument("--data_type", default='expert', type=str)
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")
    
    # Algo choice: Diffusion QL or SRPO
    parser.add_argument("--difftype", default='SRPO') # DQL for Diffusion-QL, SRPO for SRPO algo
    # JAL for joint action learning CTCE, IND for independent learning, VD for QMIX decomposition, SEQ for sequential update/regularization
    parser.add_argument("--marltype", default='SEQ') # JAL, IND, CTDE, SEQ, QMIX
    parser.add_argument("--init_actions", nargs=2, type=float, default=[0.5, -0.5]) # Initial actions as a 2D vector (e.g., -0.2 0.5)

    # Set diffusion params
    parser.add_argument("--T", default=5, type=int, help="Denoising steps for DDPM")
    parser.add_argument("--beta_schedule", default='vp', type=str)
    parser.add_argument("--seed", default=100, type=int, help="Random seed")
    parser.add_argument("--use_gpu", default=True, type=bool, help='use cuda or not')
    parser.add_argument("--device", default=0, type=int, help='cuda number')

    """   Unchangeable Params   """
    # log and save dir
    parser.add_argument("--dir", type=str, default='/home/qiaodan/Code/diffmarl/eval_result', help="tensorboard log directory")
    parser.add_argument('--dataset_dir', default='/home/qiaodan/Code/diffmarl/datasets', type=str)

    # params for MPE envs
    parser.add_argument("--discrete_action", action='store_true', default=False)
    
    # params for buffer and data
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--episode_length", default=25, type=int, help='MPE epi_length is 25, MAMuJoCo epi_length is 1000')
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)  # DDPG hidden dim
    # set_lr is unuseful
    parser.add_argument("--set_lr", action='store_true')
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--rew_scale", default=1.0, type=float)
    
    # params for RL
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--tau", default=0.01, type=float)

    # params for evaluation
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--eval_interval', default=10000, type=int)

    # training steps
    parser.add_argument('--num_steps', default=int(1e6), type=int)

    # params for logging
    parser.add_argument("--logging_interval", default=500, type=int)
    parser.add_argument("--print_interval", default=50, type=int)
    parser.add_argument("--no_log", action='store_true')

    ######### args for SRPO To be revise #########

    # regularization para
    parser.add_argument('--beta', type=float, default=None)  
    parser.add_argument('--critic_load_path', type=str, default='/home/qiaodan/Code/diffmarl/SRPO_premodels/bandit')  # HalfCheetah-v2_expert
    parser.add_argument('--diffusion_load_path', type=str, default='/home/qiaodan/Code/diffmarl/SRPO_premodels/bandit') # HalfCheetah-v2_expert
    parser.add_argument('--WT', type=str, default="VDS")
    # twin Q MLP layers
    parser.add_argument('--q_layer', type=int, default=2)
    # params for IDQL Score Network
    parser.add_argument("--actor_blocks", default=2, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--t_GassProj_dims", default=32, type=int) # 默认的是 32 dims   
    parser.add_argument("--t_embed_dims", default=64, type=int)  # 默认  64
    parser.add_argument("--sa_embed_dims", default=32, type=int)
    parser.add_argument("--resnet_hidden_dim", default=512, type=int)   # ResNet MLP hidden dim 
    parser.add_argument("--learning_rates", default=3e-4, type=float)   # lr for diffusion model
    parser.add_argument("--dilac_lr", default=3e-4, type=float)   # lr for dilac policy model
    parser.add_argument('--n_policy_epochs', default=100, type=int)
    # Dilac Policy layers, maze = 4, else = 2
    parser.add_argument('--policy_layer', type=int, default=None) 
    parser.add_argument('--regq', type=int, default=0)
    # didn't find z noise    
    # parser.add_argument('--z_noise', type=int, default=1)
    # not find
    # parser.add_argument('--critic_load_epochs', type=int, default=150)
    # regularized q gradients
    
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
    elif config.env_id == 'bandit':
        config.num_steps = int(1e3)
        config.steps_per_update = 10 # 也没用
        config.eval_interval = 100
        config.logging_interval = 100
        config.episode_length = 1
        config.gamma=0.99
        config.dilac_lr = 0.01
           
        config.tau = 0.005
        config.gamma = 0.99
    else:  # MaMujoco
        config.num_steps = int(1e6)
        config.steps_per_update = 10 # 也没用
        config.eval_interval = 5000
        config.logging_interval = 5000
        config.episode_length = 1000
        config.gamma=0.99
        config.lr = 0.0003  # 并没有进入 SRPO，只在DDPG上
           
        config.tau = 0.005
        config.gamma = 0.99

    if config.marltype == 'JAL':
        config.T=20
        # control Diffusion-QL, dont control SRPO

    if config.env_id == "bandit":
        config.dataset_dir = config.dataset_dir + '/' + config.env_id
    else:        
        config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)
        
    offline_eval(config)

