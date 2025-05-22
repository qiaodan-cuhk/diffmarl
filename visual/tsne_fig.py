import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib.colors as mcolors



class Config:
    def __init__(self):
        # 基础配置保持不变
        self.result_dir = '/data/qiaodan/code/diffmarl/results/nips/simple_spread/SEQ_random_beta0.05_critic499_seed42_2025-05-22_17-59-43-025674'
        self.quality = 'random'
        self.exp_id = 'cooperative_navigation'
        self.dataset_dir = '/data/qiaodan/code/diffmarl/datasets/simple_spread/{}/seed_2_data'.format(self.quality)
        self.title = 'Cooperative Navigation Random Dataset'
        self.plot_id = 'tsne_progress'
        self.n_agents = 3
        self.steps = [0, 25000, 50000, 75000, 100000]
        self.downsample_ratio = 0.01
        self.random_seed = 42
        self.save_dir = os.path.join('/data/qiaodan/code/diffmarl/plots', self.plot_id)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 可视化配置
        self.original_color = '#D3D3D3'  # 浅灰色
        self.progress_cmap = plt.cm.Blues  # 蓝色渐变
        self.alpha = 0.6
        self.figure_size = (12, 6)

def load_and_process_data(data_path):
    """加载数据并处理成适合t-SNE的格式"""
    data = np.load(data_path)
    all_states = []
    all_actions = []
    
    for episode_obs, episode_actions in zip(data['obs'], data['actions']):
        for step_obs, step_actions in zip(episode_obs, episode_actions):
            joint_state = np.concatenate([obs.flatten() for obs in step_obs])
            joint_action = np.concatenate([act.flatten() for act in step_actions])
            all_states.append(joint_state)
            all_actions.append(joint_action)
    
    return np.array(all_states), np.array(all_actions)

def load_dataset(dataset_dir, n_agents=3):
    """加载训练数据集"""
    all_states = []
    all_actions = []
    
    obs_data = [np.load(os.path.join(dataset_dir, f'obs_{i}.npy')) for i in range(n_agents)]
    acs_data = [np.load(os.path.join(dataset_dir, f'acs_{i}.npy')) for i in range(n_agents)]
    
    n_transitions = obs_data[0].shape[0]
    
    for step in range(n_transitions):
        joint_state = np.concatenate([obs_data[i][step].flatten() for i in range(n_agents)])
        joint_action = np.concatenate([acs_data[i][step].flatten() for i in range(n_agents)])
        all_states.append(joint_state)
        all_actions.append(joint_action)
    
    return np.array(all_states), np.array(all_actions)

def get_downsampled_dataset(dataset_combined, config):
    """对原始数据集进行降采样"""
    downsample_idx = np.random.choice(
        len(dataset_combined), 
        int(len(dataset_combined) * config.downsample_ratio), 
        replace=False
    )
    return dataset_combined[downsample_idx]

def plot_combined_tsne_distribution(original_data, eval_data_dict, save_path, title, config):
    """
    在同一个TSNE图中绘制所有数据的分布
    
    Args:
        original_data: 原始数据集
        eval_data_dict: 字典，key为step，value为对应的数据
        save_path: 保存路径
        title: 图标题
        config: 配置对象
    """
    # 将所有数据合并进行TSNE
    all_data = [original_data]
    data_labels = ['Original']
    for step, data in eval_data_dict.items():
        all_data.append(data)
        data_labels.append(f'{step}')
    
    combined_data = np.vstack(all_data)
    
    print("正在进行t-SNE降维分析...")
    tsne = TSNE(n_components=2, random_state=42)
    combined_data_2d = tsne.fit_transform(combined_data)
    
    # 分割TSNE结果
    start_idx = 0
    tsne_results = {}
    for i, label in enumerate(data_labels):
        end_idx = start_idx + len(all_data[i])
        tsne_results[label] = combined_data_2d[start_idx:end_idx]
        start_idx = end_idx
    
    # 绘图
    plt.figure(figsize=config.figure_size)
    
    # 首先绘制原始数据（灰色）
    plt.scatter(tsne_results['Original'][:, 0], tsne_results['Original'][:, 1],
               c=config.original_color, alpha=config.alpha, label='Original Dataset')
    
    # 然后绘制每个step的数据（蓝色渐变）
    n_steps = len(eval_data_dict)
    blues = plt.cm.Blues(np.linspace(0.4, 1, n_steps))  # 从浅蓝到深蓝
    
    for i, (step, data_2d) in enumerate([(k, v) for k, v in tsne_results.items() if k != 'Original']):
        plt.scatter(data_2d[:, 0], data_2d[:, 1],
                   c=[blues[i]], alpha=config.alpha, label=f'Step {step}')
    
    plt.title(title, fontsize=20)
    plt.xlabel('t-SNE Component 1', fontsize=20)
    plt.ylabel('t-SNE Component 2', fontsize=20)
    plt.legend(fontsize=14, loc='upper right', ncol=2,
              columnspacing=1.0)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存图像到: {save_path}")
    plt.close()

def main():
    config = Config()
    
    # 加载并处理原始数据集
    print("正在加载训练数据集...")
    dataset_states, dataset_actions = load_dataset(config.dataset_dir, config.n_agents)
    dataset_combined = np.concatenate([dataset_states, dataset_actions], axis=1)
    print(f"训练数据集维度: {dataset_combined.shape}")
    
    # 对原始数据集进行降采样
    downsample_dataset = get_downsampled_dataset(dataset_combined, config)
    
    # 收集所有step的数据
    eval_data_dict = {}
    for step in config.steps:
        data_path = os.path.join(config.result_dir, f'eval_data_step_{step}.npz')
        
        if not os.path.exists(data_path):
            print(f"警告：找不到时间步 {step} 的数据文件")
            continue
            
        print(f"\n处理时间步 {step} 的数据...")
        states, actions = load_and_process_data(data_path)
        eval_datasets = np.concatenate([states, actions], axis=1)
        eval_data_dict[step] = eval_datasets
        print(f"评估数据集维度: {eval_datasets.shape}")
    
    # 绘制并保存图像
    save_path = os.path.join(
        config.save_dir,
        f'{config.exp_id}_combined_progress_{config.quality}.png'
    )
    plot_combined_tsne_distribution(
        downsample_dataset,
        eval_data_dict,
        save_path,
        config.title,
        config
    )

if __name__ == "__main__":
    main()