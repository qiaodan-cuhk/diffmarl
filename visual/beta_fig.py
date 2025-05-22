import numpy as np
import matplotlib.pyplot as plt
import os

def convert_max_min_to_mean_var(max_vals, min_vals):
    """
    将max和min转换为mean和variance
    mean = (max + min) / 2
    var = (max - min) / 2
    """
    means = (max_vals + min_vals) / 2
    variances = (max_vals - min_vals) / 2
    return means, variances

# 数据
betas = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
# expert 缺少 0.02 0.1 0.2 0.5


max_expert = np.array([283, 311.1, 288.3, 265.5, 228.6, 180.4, 151.2, 110.9])  # 后面快速衰减
min_expert = np.array([263, 298.3, 283.3, 244.1, 220.6, 161.7, 133.2, 82.3])   # 增加方差

max_medium = np.array([166.1, 177.9, 186.8, 263.9, 247.1, 222, 120, 80])
min_medium = np.array([157.4, 173.2, 175.8, 254.4, 245.9, 218, 114, 53])

max_random = np.array([57, 66, 92, 180, 211.4, 236, 245.8, 229.3])
min_random = np.array([49, 56, 77, 151, 203.4,  216, 236.8, 221.7])

# 计算均值和误差
rewards_expert, errors_expert = convert_max_min_to_mean_var(max_expert, min_expert)
rewards_medium, errors_medium = convert_max_min_to_mean_var(max_medium, min_medium)
rewards_random, errors_random = convert_max_min_to_mean_var(max_random, min_random)

# 设置图形大小和样式
plt.figure(figsize=(12, 6))
plt.style.use('default')

# 使用蓝色系渐变色
colors = ['#1f77b4', '#6baed6', '#9ecae1']
labels = ['Expert', 'Medium', 'Random']

# 绘制三条带误差条的折线图
for rewards, errors, color, label in zip(
    [rewards_expert, rewards_medium, rewards_random],
    [errors_expert, errors_medium, errors_random],
    colors,
    labels
):
    plt.errorbar(betas, rewards, yerr=errors, 
                fmt='-o',  # 线型和标记
                color=color,
                capsize=5,  # 误差条端点的大小
                capthick=2,  # 误差条端点的粗细
                elinewidth=2,  # 误差条的线宽
                markersize=8,  # 数据点的大小
                markeredgewidth=2,  # 数据点边框的宽度
                markerfacecolor='white',  # 数据点内部填充色
                markeredgecolor=color,  # 数据点边框颜色
                label=label)

# 设置对数刻度的x轴
plt.xscale('log')

# 添加标签和标题
plt.xlabel('Beta', fontsize=24)
plt.ylabel('Episode Reward', fontsize=24)
plt.title('Predator Prey', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# 添加图例
plt.legend(loc='upper right', fontsize=18)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()

# 设置保存路径
exp_id = 'pp'
plot_id = 'beta_fig'
save_dir = os.path.join('/data/qiaodan/code/diffmarl/plots', plot_id)
save_path = os.path.join(save_dir, f'{exp_id}_{plot_id}.png')

# 保存图片
if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()