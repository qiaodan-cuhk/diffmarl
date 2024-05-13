# generate marl offline data in bandit

import numpy as np
import os
import argparse
from bandit import ContinuousBanditEnv
from torch.distributions import Normal
import torch
import matplotlib.pyplot as plt
from cProfile import label



env = ContinuousBanditEnv()

def select_actions():
    gaussian1 = Normal(torch.tensor([0.8, 0.8]), torch.tensor([0.1, 0.1]))
    gaussian2 = Normal(torch.tensor([-0.8, -0.8]), torch.tensor([0.1, 0.1]))

    # 随机选择一个分布，这里简单地使用0.5作为概率阈值
    if torch.rand(1).item() > 0.5:
        sample_gauss = gaussian1
    else:
        sample_gauss = gaussian2

    actions = sample_gauss.sample()
    # for action in actions:
    actions = torch.clamp(actions, env.action_space[0].low[0], env.action_space[0].high[0])
    action_np = actions.numpy()
    action_list = [np.array(action_np[0]).reshape(-1,1), np.array(action_np[1]).reshape(-1,1)]

    return action_list

def generate_data(args):

    env.seed(args.seed)
    np.random.seed(args.seed)
    
    # 收集一定数量的数据
    num_steps = int(args.data_num)  # 1e6
    agent_num = env.n_agents
    observations = []
    observations_ = []
    actions = []
    rewards = []
    dones = []

    for i in range(num_steps):
        obs = env.reset()
        # action = [env.action_space[0].sample() for _ in range(agent_num)]  # 随机选择一个动作
        action = select_actions()

        obs_next, reward, done, _ = env.step(action)
        
        observations.append(obs)
        observations_.append(obs_next)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        

    # 将数据转换为 NumPy 数组, [1e6, 5 or 2] 的 list
    observations = np.array(observations)
    observations_ = np.array(observations_)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)

    print(observations.shape)

    states = observations
    next_states = observations_

    # 定义保存数据的目录
    data_dir = args.data_save_dir
    #'/home/qiaodan/Code/diffmarl/datasets/bandit'

    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)

    # print actions distribution
    # plot
    plt.scatter(actions[:, 0], actions[:, 1], c='blue', label='Action Samples', s=1)
    plt.scatter(0.8, 0.8, color='red', marker='x', label='Mean 1', s=50)
    plt.scatter(-0.8, -0.8, color='red', marker='x', label='Mean 2', s=50)

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('2D Gaussian Distribution Data Set')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图表

    plt.savefig(os.path.join(data_dir, 'bandit_figure.png'), dpi=300, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()



    # 存储数据为 .npy 文件, 拆成两个agent存
    for id in range(agent_num):
        np.save(os.path.join(data_dir, 'obs_{}.npy'.format(id)), observations)
        np.save(os.path.join(data_dir, 'next_obs_{}.npy'.format(id)), observations_)
        np.save(os.path.join(data_dir, 'acs_{}.npy'.format(id)), actions[:, id])
        np.save(os.path.join(data_dir, 'rews_{}.npy'.format(id)), rewards)
        np.save(os.path.join(data_dir, 'dones_{}.npy'.format(id)), dones)
        np.save(os.path.join(data_dir, 'states_{}.npy'.format(id)), states)
        np.save(os.path.join(data_dir, 'next_states_{}.npy'.format(id)), next_states)

    

# 现在你可以使用 np.load 来加载数据了
# curr_obs = np.load(os.path.join(data_dir, 'obs_0.npy'))

def check(args):

    dir = args.data_save_dir
    #'/home/qiaodan/Code/diffmarl/datasets/bandit'


    for i in range(2):
        curr_obs = np.load(dir + '/' + 'obs_{}.npy'.format(i))
        curr_acs = np.load(dir + '/' + 'acs_{}.npy'.format(i))
        curr_rews = np.load(dir + '/' + 'rews_{}.npy'.format(i))
        curr_next_obs = np.load(dir + '/' + 'next_obs_{}.npy'.format(i))
        curr_dones = np.load(dir + '/' + 'dones_{}.npy'.format(i))
    
        num_experiences = curr_obs.shape[0]
        
        curr_states = np.load(dir + '/' + 'states_{}.npy'.format(i))
        curr_next_states = np.load(dir + '/' + 'next_states_{}.npy'.format(i))

        print("Current Observations shape:", curr_obs.shape)
        print("Current Actions shape:", curr_acs.shape)
        print("Current Rewards shape:", curr_rews.shape)
        print("Current Next Observations shape:", curr_next_obs.shape)
        print("Current Done flags shape:", curr_dones.shape)
        print("Current States shape:", curr_states.shape)
        print("Current Next States shape:", curr_next_states.shape)

        # 如果你想要打印所有经验的数量，可以计算 curr_obs 的第一个维度的大小
        num_experiences = curr_obs.shape[0]
        print("Number of experiences:", num_experiences)

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_num", default = 1000000, type=int)
    parser.add_argument("--seed", default = 42, type=int)
    parser.add_argument("--data_save_dir", default='/home/qiaodan/Code/diffmarl/datasets/bandit', type=str)
    
    args = parser.parse_args()

    generate_data(args)
    check(args)



