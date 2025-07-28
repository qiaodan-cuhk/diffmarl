from utils.buffer import ReplayBuffer
from multiagent_mujoco.mujoco_multi import MujocoMulti

env_args = {"scenario": "HalfCheetah-v2", "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}

env = MujocoMulti(env_args=env_args)
# args.env_args = env_args
# env.seed(args.seed + 100)
env_info = env.get_env_info()
print(env_info)
# {'state_shape': 17, 'obs_shape': 6, 'n_actions': 3, 'n_agents': 2, 'episode_limit': 1000, 'action_spaces': (Box(3,), Box(3,)), 'actions_dtype': <class 'numpy.float32'>, 'normalise_actions': False}



env_args = {"scenario": "HalfCheetah-v2", "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 1,}
env = MujocoMulti(env_args=env_args)
env_info = env.get_env_info()
print(env_info)
# {'state_shape': 17, 'obs_shape': 7, 'n_actions': 3, 'n_agents': 2, 'episode_limit': 1000, 'action_spaces': (Box(3,), Box(3,)), 'actions_dtype': <class 'numpy.float32'>, 'normalise_actions': False}


env_args = {"scenario": "Ant-v2", "episode_limit": 1000, "agent_conf": '4x2', "agent_obsk": 0,}
env = MujocoMulti(env_args=env_args)
env_info = env.get_env_info()
print(env_info)
# {'state_shape': 111, 'obs_shape': 28, 'n_actions': 3, 'n_agents': 2, 'episode_limit': 1000, 'action_spaces': (Box(3,), Box(3,)), 'actions_dtype': <class 'numpy.float32'>, 'normalise_actions': False}

env_args = {"scenario": "Ant-v2", "episode_limit": 1000, "agent_conf": '4x2', "agent_obsk": 1,}
env = MujocoMulti(env_args=env_args)
env_info = env.get_env_info()
print(env_info)
# {'state_shape': 111, 'obs_shape': 31, 'n_actions': 3, 'n_agents': 2, 'episode_limit': 1000, 'action_spaces': (Box(3,), Box(3,)), 'actions_dtype': <class 'numpy.float32'>, 'normalise_actions': False}





# replay_buffer = ReplayBuffer(int(1e6),
#                              2,
#                                      [env_info['obs_shape'] for _ in env.observation_space],
#                                      [acsp.shape[0] for acsp in env.action_space],
#                                      is_mamujoco=True,
#                                      state_dims=[env_info['state_shape'] for _ in env.observation_space],
#                                      device = 'cpu')


# dataset_dir = '/data/qiaodan/code/diffmarl/datasets/HalfCheetah-v2/expert/seed_0_data'
# replay_buffer.load_batch_data(dataset_dir, rew_scale=1.0)

# print(replay_buffer.obs_buffs[0].shape)   # (1000000, 6) 不同agent不同
# print(replay_buffer.obs_buffs[1].shape)  # (1000000, 6)
# print(replay_buffer.ac_buffs[0].shape)   # (1000000, 3)
# print(replay_buffer.ac_buffs[1].shape)   # (1000000, 3)
# print(replay_buffer.rew_buffs[0].shape)  # (1000000,)
# print(replay_buffer.rew_buffs[1].shape)  # (1000000,)

# print(replay_buffer.state_buffs[0].shape)  # (1000000, 17)  不同agent相等
# print(replay_buffer.state_buffs[1].shape)  # (1000000, 17)


