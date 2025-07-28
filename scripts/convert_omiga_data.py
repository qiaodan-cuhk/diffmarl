import numpy as np
import h5py
import os
import argparse

# === OMIGA环境与omar不同 === 


def convert_omiga_to_diffmarl(data_dir, env_name, quality, output_dir):
    """
    基于OMIGA的load方法，直接转换数据格式
    """
    print('==========Data loading==========')
    data_file = data_dir + env_name + quality + '.hdf5'
    print('Loading from:', data_file)
    
    # 使用OMIGA的加载逻辑
    f = h5py.File(data_file, 'r')
    s = np.array(f['s'])   # s与o一致，1001000, 6, 23
    o = np.array(f['o'])
    a = np.array(f['a'])   # 1001000, 6, 1
    r = np.array(f['r'])   # 1001000, 1
    d = np.array(f['d'])   # d.sum=1000,每一条存了1001个，后边会处理
    f.close()

    env_args = {"scenario": "HalfCheetah-v2", "episode_limit": 1000, "agent_conf": '6x1', "agent_obsk": 1,}
    from multiagent_mujoco.mujoco_multi import MujocoMulti
    env = MujocoMulti(env_args=env_args)
    # args.env_args = env_args
    # env.seed(args.seed + 100)
    env_info = env.get_env_info()
    # 返回的是17 state和 2 obs
    #     env.get_obs()
    # [array([0.04459093, 0.08326562]), array([-0.02780823, -0.09668242]), array([0.05055583, 0.01252794]), array([0.02501438, 0.0360668 ]), array([-0.09835785,  0.01952816]), array([ 0.01833596, -0.02389472])]
    # env.get_state()
    # array([ 0.04605756,  0.04113704,  0.05055583, -0.02780823,  0.04459093,
    #         0.01833596, -0.09835785,  0.02501438,  0.03722591,  0.0366072 ,
    #         0.1020495 ,  0.01252794, -0.09668242,  0.08326562, -0.02389472,
    #         0.01952816,  0.0360668 ])

    data_size = s.shape[0]
    nonterminal_steps, = np.where(
        np.logical_and(
            np.logical_not(d[:,0]),
            np.arange(data_size) < data_size - 1))
    print('Found %d non-terminal steps out of a total of %d steps.' % (
        len(nonterminal_steps), data_size))

    # 使用OMIGA的处理逻辑
    save_o = o[nonterminal_steps]
    save_s = s[nonterminal_steps]
    save_a = a[nonterminal_steps]
    save_r = r[nonterminal_steps]
    save_mask = 1 - d[nonterminal_steps + 1]
    save_s_next = s[nonterminal_steps + 1]
    save_o_next = o[nonterminal_steps + 1]
    a_next = a[nonterminal_steps + 1]
    
    print(f"处理后数据形状: s={save_s.shape}, o={save_o.shape}, a={save_a.shape}")
    
    # omiga 采取了obsk=1，利用了邻居的信息，但是从数据集加载来看，obs=state且23dim，与当前的17dim和2dim obs不一致，reward倒是一致
    
    # output_dir = output_dir + '/6_halfcheetah/' + quality

    # # 创建输出目录
    # os.makedirs(output_dir, exist_ok=True)
    
    # # 按智能体分离数据并保存
    # n_agents = s.shape[1]
    # for i in range(n_agents):
    #     print(f"保存智能体 {i} 的数据...")
        
    #     # 直接保存为DiffMARL格式
    #     np.save(os.path.join(output_dir, f'obs_{i}.npy'), o[:, i, :])
    #     np.save(os.path.join(output_dir, f'acs_{i}.npy'), a[:, i, :])
    #     np.save(os.path.join(output_dir, f'rews_{i}.npy'), r[:, i])
    #     np.save(os.path.join(output_dir, f'next_obs_{i}.npy'), o_next[:, i, :])
    #     np.save(os.path.join(output_dir, f'dones_{i}.npy'), mask[:, i])
    #     np.save(os.path.join(output_dir, f'states_{i}.npy'), s[:, i, :])
    #     np.save(os.path.join(output_dir, f'next_states_{i}.npy'), s_next[:, i, :])
    
    # print(f"转换完成! 数据已保存到 {output_dir}")
    # print(f"每个智能体有 {s.shape[0]} 个样本")

def main():
    parser = argparse.ArgumentParser(description='OMIGA到DiffMARL数据格式转换')
    parser.add_argument('--data_dir', type=str, default='/data/qiaodan/downloads/omiga/', help='HDF5数据目录')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2-6x1-', help='环境名称')
    parser.add_argument('--quality', type=str, default='', help='数据质量')
    parser.add_argument('--output_dir', type=str, default='/data/qiaodan/code/diffmarl/datasets/omiga', help='输出目录')
    
    args = parser.parse_args()
    
    convert_omiga_to_diffmarl(args.data_dir, args.env_name, args.quality, args.output_dir)

if __name__ == "__main__":
    main()