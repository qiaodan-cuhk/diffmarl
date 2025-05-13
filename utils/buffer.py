import numpy as np
from torch import Tensor
from torch.autograd import Variable
import torch
import math

# halfcheetah medium replay 46w 数据
# simple spread mid replay 97500
# simple tag mid replay 62500
# simple world mid replay 80000


class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, device, is_mamujoco=False, state_dims=None):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))

        self.is_mamujoco = is_mamujoco
        if self.is_mamujoco:
            self.state_buffs = []
            self.next_state_buffs = []
            for sdim in state_dims:
                self.state_buffs.append(np.zeros((max_steps, sdim)))
                self.next_state_buffs.append(np.zeros((max_steps, sdim)))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)
        self.device = device

    def __len__(self):
        return self.filled_i

    def sample(self, N, to_gpu=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)  # 默认是false，但用true可以加速大数据集的采样效率
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).to(self.device)
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cpu()

        bf = []

        if self.is_mamujoco:
            for i in range(self.num_agents):                
                agent_data = {"state": cast(self.state_buffs[i][inds]),
                              "obs": cast(self.obs_buffs[i][inds]),
                              "action": cast(self.ac_buffs[i][inds]),
                              "rewards": cast(self.rew_buffs[i][inds]),
                              "next_state": cast(self.next_state_buffs[i][inds]),
                              "next_obs": cast(self.next_obs_buffs[i][inds]),
                              "done": cast(self.done_buffs[i][inds]),
                              "next_action": cast(self.ac_buffs[i][(inds+1)%self.filled_i])
                              }

                bf.append(agent_data)
        else:
            for i in range(self.num_agents):
                agent_data = {"obs": cast(self.obs_buffs[i][inds]),
                              "action": cast(self.ac_buffs[i][inds]),
                              "rewards": cast(self.rew_buffs[i][inds]),
                              "next_obs": cast(self.next_obs_buffs[i][inds]),
                              "done": cast(self.done_buffs[i][inds]),
                              "next_action": cast(self.ac_buffs[i][(inds+1)%self.filled_i])
                              }

                bf.append(agent_data)

        # data structure: [{1} {2} ... {N}]

        return bf

    def load_batch_data(self, dir, rew_scale=1.0):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        all_min_rews = []
        for i in range(self.num_agents):
            curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
            curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
            curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
            curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
            curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))

            if "bandit" in dir:
                curr_acs = curr_acs.reshape(-1,1)

            num_experiences = curr_obs.shape[0]    # 数据数量

            # random_indices = np.random.choice(int(100000), size=self.max_steps, replace=False)
            self.obs_buffs[i][:num_experiences] = curr_obs
            self.ac_buffs[i][:num_experiences] = curr_acs
            self.rew_buffs[i][:num_experiences] = curr_rews * rew_scale
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs
            # self.done_buffs[i][:num_experiences] = curr_dones    # 要修改mpe的原始数据增加done=1

            if self.is_mamujoco:
                self.done_buffs[i][:num_experiences] = curr_dones
                self.ave_reward = np.sum(self.rew_buffs[i][:num_experiences]) / np.sum(self.done_buffs[i][:num_experiences])
                self.sum_reward = np.sum(self.rew_buffs[i][:num_experiences])
                # mamujoco 额外状态信息
                curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
                curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))
                self.state_buffs[i][:num_experiences] = curr_states
                self.next_state_buffs[i][:num_experiences] = curr_next_states
            elif "bandit" in dir:
                self.done_buffs[i][:num_experiences] = curr_dones
            else:
                episode_length = 25
                steps = np.arange(num_experiences)
                modified_done = (steps % episode_length == episode_length-1).astype(np.float32)
                # self.done_buffs[i][:num_experiences] = np.maximum(curr_dones, modified_done)  # 原来的方案
                self.done_buffs[i][:num_experiences] = modified_done  # 直接覆盖方案

                self.ave_reward = np.sum(self.rew_buffs[i][:num_experiences]) / (num_experiences/episode_length)  # 根据main.py, MPE episode length eval is 25
                self.sum_reward = np.sum(self.rew_buffs[i][:num_experiences])
         
        self.filled_i = num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences

    # 用于mamujoco 210数据集格式from ogmarl
    def load_batch_data_new_format(self, dir, rew_scale=1.0):
        """
        加载新格式的数据，数据格式为：
        - obs.npy: 所有智能体的观察空间
        - actions.npy: 所有智能体的动作
        - rewards.npy: 所有智能体的奖励
        - path_lengths.npy: 轨迹长度，用于确定episode结束
        """
        print('\033[1;33mloading new format batch data from {}...\033[1;0m'.format(dir))
        
        # 加载数据
        observations = np.load(os.path.join(dir, 'obs.npy'))
        actions = np.load(os.path.join(dir, 'actions.npy'))
        rewards = np.load(os.path.join(dir, 'rewards.npy'))
        path_lengths = np.load(os.path.join(dir, 'path_lengths.npy'))
        
        # 计算总样本数
        num_experiences = observations.shape[0]
        
        # 根据path_lengths生成done信号
        dones = np.zeros_like(rewards)  # 与rewards同形状
        start = 0
        for path_length in path_lengths:
            dones[start + path_length - 1] = 1  # 每个轨迹的最后一步标记为done
            start += path_length
        
        # 生成next_obs（通过移位）
        next_observations = np.roll(observations, -1, axis=0)
        # 对于每个轨迹的最后一步，其next_obs需要特殊处理
        start = 0
        for path_length in path_lengths:
            if start + path_length < num_experiences:
                next_observations[start + path_length - 1] = observations[start + path_length]
            start += path_length
        
        # 为每个智能体分配数据
        for i in range(self.num_agents):
            # 假设数据的第二维是智能体维度
            self.obs_buffs[i][:num_experiences] = observations[:, i]
            self.ac_buffs[i][:num_experiences] = actions[:, i]
            self.rew_buffs[i][:num_experiences] = rewards[:, i] * rew_scale
            self.next_obs_buffs[i][:num_experiences] = next_observations[:, i]
            self.done_buffs[i][:num_experiences] = dones[:, i]
            
            if self.is_mamujoco:
                # 如果需要额外的状态信息，可以从observations中提取或加载额外的文件
                self.state_buffs[i][:num_experiences] = observations[:, i]  # 可能需要调整
                self.next_state_buffs[i][:num_experiences] = next_observations[:, i]  # 可能需要调整
                
                # 计算平均奖励和总奖励
                self.ave_reward = np.sum(self.rew_buffs[i][:num_experiences]) / len(path_lengths)
                self.sum_reward = np.sum(self.rew_buffs[i][:num_experiences])
        
        self.filled_i = num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences