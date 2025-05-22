import numpy as np
import matplotlib.pyplot as plt
import os

# 数据
betas = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])

# 这里需要替换成您的实际数据
# 示例数据（请替换成您的真实数据）
rewards = np.array([100, 120, 150, 140, 130, 110, 90, 70])  # 平均奖励
errors = np.array([10, 12, 15, 14, 13, 11, 9, 7])  # 标准差或标准误

# 设置图形大小和样式
plt.figure(figsize=(12, 6))
plt.style.use('default')

# 使用与原代码相同的蓝色
color = '#1f77b4'

# 绘制带误差条的折线图
plt.errorbar(betas, rewards, yerr=errors, 
             fmt='-o',  # 线型和标记
             color=color,
             capsize=5,  # 误差条端点的大小
             capthick=2,  # 误差条端点的粗细
             elinewidth=2,  # 误差条的线宽
             markersize=8,  # 数据点的大小
             markeredgewidth=2,  # 数据点边框的宽度
             markerfacecolor='white',  # 数据点内部填充色
             markeredgecolor=color)  # 数据点边框颜色

# 设置对数刻度的x轴
plt.xscale('log')

# 添加标签和标题
plt.ylabel('Episode Reward', fontsize=24)
plt.title('Ablation Study: Beta', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()

# 设置保存路径
exp_id = 'Beta_Ablation'
plot_id = 'ablation_study'
save_dir = os.path.join('/data/qiaodan/code/diffmarl/plots', plot_id)
save_path = os.path.join(save_dir, f'{exp_id}.png')

# 保存图片
if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()