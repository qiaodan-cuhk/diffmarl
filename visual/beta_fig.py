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

# Simple Spread Scores
# max_expert = np.array([531, 484.4, 449.2, 386, 166.7, 174.4, 164.8, 133.7])  # 后面快速衰减
# min_expert = np.array([528.9, 483.3, 448.7, 366.1, 164.1, 167.6, 146, 131.9])   # 增加方差

# max_medium = np.array([271.9, 266.2, 281.7, 282.3, 199.3, 154.8, 97.6, 87])
# min_medium = np.array([261.6, 264, 280.2, 277.4, 196.4, 135, 90.4, 83])

# max_random = np.array([213, 197.5, 293, 383, 403, 405, 406, 417])
# min_random = np.array([193, 177.8, 273, 347, 387, 389, 398, 397])


# Simple Tag Scores
# max_expert = np.array([283, 311.1, 288.3, 265.5, 228.6, 180.4, 151.2, 110.9])  # 后面快速衰减
# min_expert = np.array([263, 298.3, 283.3, 244.1, 220.6, 161.7, 133.2, 82.3])   # 增加方差

# max_medium = np.array([160, 180, 200, 190, 180, 160, 140, 120])
# min_medium = np.array([140, 160, 180, 170, 160, 140, 120, 100])

# max_random = np.array([110, 130, 150, 140, 130, 110, 90, 70])
# min_random = np.array([90, 110, 130, 120, 110, 90, 70, 50])


# Simple World Scores
max_expert = np.array([131.5, 126.5, 137.6, 128.7, 137.1, 113.5, 98.2, 91.1]) 
min_expert = np.array([124.5, 119.5, 133.6, 124.1, 124.5, 106.5, 91.3, 75.8])   

# 缺少前2后3
max_medium = np.array([98.3, 94.4, 109.4, 107.9, 143.6, 103.9, 82.8, 79.6])
min_medium = np.array([76.3, 84.1, 94.4, 100.9, 127.6, 88.3, 73.4, 63.1])

max_random = np.array([-2.24, -4.1, -3.9, -0.1, 22, 97, 124, 119.6])
min_random = np.array([-3.9, -4.7, -4.5, -4.8, 11.9, 72, 104.4, 103.1])

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
plt.title('World', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# 添加图例
plt.legend(loc='upper right', fontsize=18)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()

# 设置保存路径
exp_id = 'world'
plot_id = 'beta_fig'
save_dir = os.path.join('/data/qiaodan/code/diffmarl/plots', plot_id)
save_path = os.path.join(save_dir, f'{exp_id}_{plot_id}.png')

# 保存图片
if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')

plt.show()