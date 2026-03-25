""" This file is used for load pretrained diffusion models and critic models and evaluate their performance """

# used for visualize bandit trajectory


import numpy as np
import torch
import gym





def simple_eval_policy(policy_fn, env_name, seed, eval_episodes=20):
    env = gym.make(env_name)
    env.seed(seed+561)
    all_rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()
        total_reward = 0.
        done = False
        while not done:
            with torch.no_grad():
                action = policy_fn(torch.Tensor(obs).unsqueeze(0).to("cuda")).cpu().numpy().squeeze()
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
            else:
                obs = next_obs
        all_rewards.append(d4rl.get_normalized_score(env_name, total_reward))
    return np.mean(all_rewards), np.std(all_rewards)

def pallaral_simple_eval_policy(policy_fn, env_name, seed, eval_episodes=20):
    eval_envs = []
    for i in range(eval_episodes):
        env = gym.make(env_name)
        eval_envs.append(env)
        env.seed(seed + 1001 + i)
        env.buffer_state = env.reset()
        env.buffer_return = 0.0
    ori_eval_envs = [env for env in eval_envs]
    import time
    t = time.time()
    while len(eval_envs) > 0:
        new_eval_envs = []
        states = np.stack([env.buffer_state for env in eval_envs])
        states = torch.Tensor(states).to("cuda")
        with torch.no_grad():
            actions = policy_fn(states).detach().cpu().numpy()
        for i, env in enumerate(eval_envs):
            state, reward, done, info = env.step(actions[i])
            env.buffer_return += reward
            env.buffer_state = state
            if not done:
                new_eval_envs.append(env)
        eval_envs = new_eval_envs
    for i in range(eval_episodes):
        ori_eval_envs[i].buffer_return = d4rl.get_normalized_score(env_name, ori_eval_envs[i].buffer_return)
    mean = np.mean([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
    std = np.std([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
    return mean, std







