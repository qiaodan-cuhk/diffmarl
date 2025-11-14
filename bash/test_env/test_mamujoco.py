from multiagent_mujoco.mujoco_multi import MujocoMulti
import numpy as np
import time



# state dim 17, obs dim 13 if global categories, obs dim 7 if obsk 1, obs dim 6 if obsk 0

def main():
    env_args = {
            "scenario": "HalfCheetah-v2",
            "episode_limit": 1000,
            "agent_conf": "2x3",
            "agent_obsk": 0,
            # "global_categories": "qvel,qpos",
        }
    # env_args = {"scenario": "HalfCheetah-v2",
    #               "agent_conf": "2x3",
    #               "agent_obsk": 0,
    #               "episode_limit": 1000}
    env = MujocoMulti(env_args=env_args)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.uniform(-1.0, 1.0, n_actions)
                actions.append(action)

            # print(env.step(actions))
            next_obs, reward, terminated, info = env.step(actions)

            print(f"state shape: {state.shape}")
            print(f"next_obs shape: {next_obs.shape}")
            print(f"reward: {reward}")
            print(f"terminated: {terminated}")
            print(f"info: {info}")

            terminated = terminated[0]
            episode_reward += reward

            time.sleep(0.1)
            # env.render()


        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()

if __name__ == "__main__":
    main()