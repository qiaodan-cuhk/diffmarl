import numpy as np
from torch import Tensor
from torch.autograd import Variable
import torch
import math
import h5py
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
        # mamujoco加载state，但是训练critic和behavior都使用obs
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


    def load_batch_data_omiga(self, dir, rew_scale=1.0):
        print ('\033[1;33mloading batch data from {}...\033[1;0m'.format(dir))
        # dir = /data/qiaodan/code/diffmarl/datasets/omiga/HalfCheetah-v2-6x1-expert.hdf5

        f = h5py.File(dir, 'r')
        s = np.array(f['s'])
        o = np.array(f['o'])
        a = np.array(f['a'])
        r = np.array(f['r'])
        d = np.array(f['d'])
        f.close()

        # 只保留非终止步
        data_size = s.shape[0]
        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(d[:,0]),
                np.arange(data_size) < data_size - 1))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), data_size))



        curr_obs = o[nonterminal_steps]
        curr_states = s[nonterminal_steps]  # [1e6, 6, 23]
        curr_acs = a[nonterminal_steps]     # [1e6, 6, 1]
        curr_rews = r[nonterminal_steps].reshape(-1, 1)            # [1e6,1]
        curr_dones = d[nonterminal_steps + 1].reshape(-1, 1)   # [1e6,1]
        curr_next_obs = o[nonterminal_steps + 1]
        curr_next_states = s[nonterminal_steps + 1]
        # curr_next_acs = a[nonterminal_steps + 1]


        num_experiences = curr_obs.shape[0] # 数据数量




        for i in range(self.num_agents):



            self.obs_buffs[i][:num_experiences] = curr_obs[:,i,:]
            self.ac_buffs[i][:num_experiences] = curr_acs[:,i,:]
            self.rew_buffs[i][:num_experiences] = curr_rews.flatten() * rew_scale
            self.next_obs_buffs[i][:num_experiences] = curr_next_obs[:,i,:]

            if self.is_mamujoco:
                self.done_buffs[i][:num_experiences] = curr_dones.flatten()
                self.ave_reward = np.sum(self.rew_buffs[i][:num_experiences]) / np.sum(self.done_buffs[i][:num_experiences])
                self.sum_reward = np.sum(self.rew_buffs[i][:num_experiences])
                # mamujoco 额外状态信息，但实际上训练并不需要state
                self.state_buffs[i][:num_experiences] = curr_states[:,i,:]
                self.next_state_buffs[i][:num_experiences] = curr_next_states[:,i,:]
         
        self.filled_i = num_experiences
        self.curr_i = 0 if self.curr_i == self.max_steps else num_experiences
