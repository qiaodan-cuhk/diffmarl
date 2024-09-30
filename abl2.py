import numpy as np
import matplotlib.pyplot as plt

# 示例数据
# x = np.arange(1, 6)  # x 轴数据
# mean_values = np.array([2.5, 3.0, 3.5, 4.0, 4.5])  # 平均值
# std_dev = np.array([0.2, 0.3, 0.1, 0.4, 0.2])  # 标准差作为误差


etas = np.array([0.001, 0.002, 0.005, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5])

ctde_exp_max = np.array([0, 3830, 3780, 3472, 346.8, 0, 0, 0, -350.5, 0])
ind_exp_max = np.array([0, 0, 0, 0, 3648, 0, 0, 0, 0, 0])
seq_exp_max = np.array([3681, 0, 3945, 3624, 1872, 0, 0, 0, 0, 0])

ctde_exp_min = np.array([0, 3802, 3551, 3353, 147.3, 0, 0, 0, -370.6, 0])
ind_exp_min = np.array([0, 0, 0, 0, 3543, 0, 0, 0, 0, 0])
seq_exp_min = np.array([3627, 0, 3844, 3447, 1710, 0, 0, 0, 0, 0])

ctde_exp_max_normal = ctde_exp_max / np.max(ctde_exp_max)
ind_exp_max_normal = ind_exp_max / np.max(ind_exp_max)
seq_exp_max_normal = seq_exp_max / np.max(seq_exp_max)

ctde_exp_min_normal = ctde_exp_min / np.max(ctde_exp_max)
ind_exp_min_normal = ind_exp_min / np.max(ind_exp_max)
seq_exp_min_normal = seq_exp_min / np.max(seq_exp_max)

# 计算均值和误差
ctde_mean = (ctde_exp_max_normal + ctde_exp_min_normal) / 2
ctde_error = (ctde_exp_max_normal - ctde_exp_min_normal) / 2

ind_mean = (ind_exp_max_normal + ind_exp_min_normal) / 2
ind_error = (ind_exp_max_normal - ind_exp_min_normal) / 2

seq_mean = (seq_exp_max_normal + seq_exp_min_normal) / 2
seq_error = (seq_exp_max_normal - seq_exp_min_normal) / 2

# 创建图形
plt.figure(figsize=(10, 6), dpi=300)


# 绘制带误差线的折线图
plt.errorbar(etas[:len(ctde_mean)], ctde_mean, yerr=ctde_error, fmt='-o', label='CTDE', capsize=5)
plt.errorbar(etas[:len(ind_mean)], ind_mean, yerr=ind_error, fmt='-o', label='IND', capsize=5)
plt.errorbar(etas[:len(seq_mean)], seq_mean, yerr=seq_error, fmt='-o', label='SEQ', capsize=5)

# 设置 x 轴为对数刻度
plt.xscale('log')


# 添加标签和标题
plt.xlabel('Learning Rate (eta)')
plt.ylabel('Performance')
plt.title('Ablation Study: Effect of Hyperparameters on Performance')
plt.legend()

# 禁用网格
plt.grid(False)
# 自动调整布局
plt.tight_layout()


# 保存图形
plt.savefig('abl2_exp.png')
plt.show()

# ctde_med_max
# ind_med_max
# seq_med_max

# ctde_med_min
# ind_med_min
# seq_med_min

# ctde_med_rep_max
# ind_med_rep_max
# seq_med_rep_max

# ctde_med_rep_min
# ind_med_rep_min
# seq_med_rep_min

# ctde_rand_max
# ind_rand_max
# seq_rand_max

# ctde_rand_min
# ind_rand_min
# seq_rand_min

