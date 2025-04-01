""" Pretrain joint Q(s,a) or advantage Q(s, a-i, ai) """
# 需要确认的：
# 1.添加env id选项，处理数据合并；
# 2.确认mpe的另外两个任务，prey在不在数据集，以及obs/state；
# 3.完善jal的数据合并，debug检查
# Line 335 需要增加 2ant 4ant env info
# Line 125 的 next action 似乎没用，数据集也没有


import os
import numpy as np
import torch
import tqdm
import argparse
import datetime
from gym.spaces import Box, Discrete
from utils.buffer import ReplayBuffer
from utils.make_env import make_env
from utils.env_wrappers import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter\


# IQL algo
from algorithms.SRPO import MASRPO_IQL
from utils.agents import DDPGAgent

# MAMujoco
try:
    from multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('MujocoMulti not installed')
# bandit
from bandit import ContinuousBanditEnv
# make parallel MA-Env for MPE
def make_parallel_env(env_id, seed, discrete_action):
    def get_env_fn(rank):
        env = make_env(env_id, discrete_action=discrete_action)
        env.seed(seed + rank * 1000)
        np.random.seed(seed + rank * 1000)
        return env
    return DummyVecEnv([get_env_fn(0)])

# 暂时只考虑JAL的eval 还没写完prey的逻辑
def eval_policy(agent, env_name, seed, eval_episodes, discrete_action, device='cpu', env_args=None):
    if env_name in ['HalfCheetah-v2']:
        env = MujocoMulti(env_args=env_args)
        env.seed(seed + 100)
        all_episodes_rewards = []
        for ep_i in range(eval_episodes):
            env.reset()
            done = False
            episode_reward = 0.
            while not done:
                obs = env.get_obs()
                torch_obs = [torch.Tensor(obs[i]).unsqueeze(0).to(device) for i in range(len(obs))]
                concat_obs = torch.cat(torch_obs, dim=1)
                # concat因为这是joint action IQL

                actions = agent.deter_policy.select_actions(concat_obs)  # [n]


                if torch.is_tensor(actions):
                    actions = actions.cpu().numpy()
                # 解开concatenated动作
                split_actions = np.split(actions, len(obs), axis=1)
                # 执行动作
                reward, done, info = env.step([a.squeeze(0) for a in split_actions])
                episode_reward += reward

            all_episodes_rewards.append(episode_reward)        
        mean_episode_reward = np.mean(np.array(all_episodes_rewards))
        return mean_episode_reward
    elif env_name in ['simple_spread', 'simple_tag', 'simple_world']:
        avg_predator_return = 0.
        env = make_parallel_env(env_name, seed + 100, discrete_action)

        for ep_i in range(0, eval_episodes):
            obs = env.reset()    # 修改过，返回的是list不再是array
            obs = np.array(obs)  # 处理返回的list形态reset数据

            # agent.prep_rollouts(device=device)    # 这个东西未必有用in IQL

            for et_i in range(25):   # mpe固定长度25
                obs_len = args.predator_nums
                """这里agent没有num preys这个属性"""
                # if env_name in ['simple_tag', 'simple_world']:  # if predator-prey
                #     obs_len += agent.num_preys

                torch_obs_agent = torch.Tensor(obs[:,:obs_len].reshape(obs.shape[0], -1)).to(device)
                # torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(obs_len)]

                agent.deter_policy.to(device)   # 把模型挪到cpu上
                torch_agent_actions = agent.deter_policy.select_actions(torch_obs_agent)
                # split为三个agent
                if torch.is_tensor(torch_agent_actions):
                    numpy_agent_actions = torch_agent_actions.detach().cpu().numpy()
                # 解开concatenated动作
                split_actions = np.split(numpy_agent_actions, obs_len, axis=1)

                split_actions = [action.squeeze() for action in split_actions]
                # 从 tensor([[a], [a], [a]]) 变为 list[np[], np[], np[]]
                # 最内层action必须是 [2]，不可以是[1,2]，否则在env.step中会报维度错误


                """还没测试这部分"""
                if env_name in ['simple_tag', 'simple_world']:  # 需要额外加载prey的策略和选择动作
                    # 原始的MASRPO中的prey加载方式
                    # prey_agents = [DDPGAgent(lr=lr, discrete_action=self.discrete_action, hidden_dim=self.hidden_dim, **params) for params in adv_init_params]
                    # 为了debug暂时设置的
                    prey_agents = []
                    prey_agents.load_policy()
                    torch_obs_prey = torch.Tensor(obs[:,obs_len:].reshape(obs.shape[0], -1)).to(device)
                    prey_actions = prey_agents.select_actions(torch_obs_prey)
                    actions = [split_actions, prey_actions]
                else:
                    actions = [split_actions]

                next_obs, rewards, dones, infos = env.step(actions)
                
                if env_name in ['simple_tag', 'simple_world']:
                    avg_predator_return += rewards[0][0]
                else:
                    avg_agent_reward = np.mean(rewards[0])
                    avg_predator_return += avg_agent_reward


                # 这里修改了MPE的原函数 utils/env_wrappers，返回的next obs是一个list而不是array
                next_obs = np.array(next_obs)

                obs = next_obs
                

        avg_predator_return /= eval_episodes
        return avg_predator_return
    else:
        print("Wrong Env in Eval Policy")
        return 0

    

"""目前仅 joint critic 增加了grad norm记录"""
def get_grad_norm(parameters):
    """计算梯度范数的辅助函数"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return np.sqrt(total_norm)


# Q(s, a_i)
def train_ind_critic(args, score_model, data_loader, agent_num, writer, env_id, start_epoch=0):
    n_epochs = args.training_epoch   # 200
    evaluation_inerval = args.eval_interval   # 5
    epoch_save_interval = args.save_interval  # 20

    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    best_loss = 1e5

    for epoch in tqdm_epoch:
        avg_critic_loss = 0.
        avg_bc_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size, to_gpu=args.use_gpu)  # 如果用gpu, True
            data_i = data[agent_num]
            loss_policy, loss_bc = score_model.update_iql(data_i)
            avg_critic_loss += loss_policy.detach().cpu().numpy()
            avg_bc_loss += loss_bc.detach().cpu().numpy()
            num_items += 1
            writer.add_scalar('agent {}/episode critic loss'.format(agent_num), loss_policy, step_in_epoch)
            writer.add_scalar('agent {}/episode BC loss'.format(agent_num), loss_bc, step_in_epoch)
            
        tqdm_epoch.set_description('Critic Loss Agent {}: {:5f}'.format(agent_num, avg_critic_loss / num_items))
        
        epoch_loss = score_model.policy_loss.detach().cpu().numpy()

        """ Logging """
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:

            # 正常是要保留这个eval环境验证，记录eval reward来评估表现的
            # if (epoch % 5 == 4) or epoch==0:
                # eval_return = eval_policy(ma_agent, config.env_id, config.seed, config.eval_episodes, config.discrete_action, device='cpu', env_args=env_args)
                # mean, std = pallaral_simple_eval_policy(score_model.deter_policy.select_actions,args.env,00)
                # args.run.log({"eval/rew{}".format("deter"): mean}, step=epoch+1)

            writer.add_scalar("agent {}/v_loss".format(agent_num), score_model.q[0].v_loss.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("agent {}/q_loss".format(agent_num), score_model.q[0].q_loss.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("agent {}/q".format(agent_num), score_model.q[0].q.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("agent {}/v".format(agent_num), score_model.q[0].v.detach().cpu().numpy(), epoch+1)
            # policy loss = bc_loss * exp(Q-V)
            writer.add_scalar("agent {}/policy_loss".format(agent_num), score_model.policy_loss.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("agent {}/mean policy loss".format(agent_num), avg_critic_loss / num_items, epoch+1)
            writer.add_scalar("agent {}/mean bc loss".format(agent_num), avg_bc_loss / num_items, epoch+1)
            # policy loss 用了 cosineAnnealing 150w steps
            writer.add_scalar("agent {}/lr".format(agent_num), score_model.deter_policy_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)
            
        
        """ Save models """
        if args.save_model and epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New lowest critic loss in epoch {}, Save models".format(epoch))
            torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", "IND", "best_critic_{}.pth".format(agent_num)))
            # SRPO_premodels/env_id/IND/best_critic_i.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            print("Save critic models: Epoch {}".format(epoch))
            torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", "IND", "critic_{}_epoch{}.pth".format(agent_num, epoch)))
            # SRPO_premodels/env_id_level/IND/critic_1_epoch150.pth



# MPE 的 JAL Q 需要修改
def train_joint_critic(args, score_model, data_loader, writer, env_id, start_epoch=0):
    n_epochs = args.training_epoch   # 200
    evaluation_inerval = args.eval_interval   # 5
    epoch_save_interval = args.save_interval  # 20

    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    best_loss = 1e5

    # 用于加速验证，提前到这里，后边可以删掉
    # mean_episode_reward = eval_policy(score_model, args.env_id, args.seed, 10, args.discrete_action, device='cpu', env_args=args.env_args)

    for epoch in tqdm_epoch:
        avg_critic_loss = 0.
        avg_bc_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size, to_gpu=args.use_gpu)
            
            # we need to concate marl datasets with [{}, {}] into one dict
            # used for 2 halfcheetah
            if env_id in ['HalfCheetah-v2', '2-ant']:   # '4-ant'
                d0 = data[0]
                d1 = data[1]
                data_concate = {}
                for item in ["obs", "action", "next_obs", "next_action"]:
                    data_concate[item] = torch.cat((d0[item], d1[item]), dim=1).to(args.device)
                    assert data_concate[item].size()[0] == args.batch_size
                # 以下内容不需要concate
                data_concate["state"] = d0["state"].to(args.device)
                data_concate["next_state"] = d0["next_state"].to(args.device)
                data_concate["rewards"] = d0["rewards"].to(args.device)
                data_concate["done"] = d0["done"].to(args.device)
            elif env_id in ['simple_spread', 'simple_tag', 'simple_world']:
                # Only concatenate data for the first num_agents as predator args.predator_nums
                data_concate = {
                    "obs": torch.stack([data[i]["obs"] for i in range(args.predator_nums)], dim=1).reshape(args.batch_size, -1).to(args.device),
                    "action": torch.stack([data[i]["action"] for i in range(args.predator_nums)], dim=1).reshape(args.batch_size, -1).to(args.device), 
                    "next_obs": torch.stack([data[i]["next_obs"] for i in range(args.predator_nums)], dim=1).reshape(args.batch_size, -1).to(args.device),
                    "rewards": data[0]["rewards"].to(args.device),
                    "done": data[0]["done"].to(args.device)
                }

            loss_policy, loss_bc = score_model.update_iql(data_concate)

            
            avg_critic_loss += loss_policy.detach().cpu().numpy()
            avg_bc_loss += loss_bc.detach().cpu().numpy()
            num_items += 1
            writer.add_scalar('JAL/episode critic loss', loss_policy, step_in_epoch)
            writer.add_scalar('JAL/episode BC loss', loss_bc, step_in_epoch)

        tqdm_epoch.set_description('Average Critic Loss: {:5f}'.format(avg_critic_loss / num_items))
        
        epoch_loss = score_model.policy_loss.detach().cpu().numpy()

        """ Log by tensorboard """
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            if (epoch % 5 == 4) or epoch==0:
                # mean, std = eval_policy(score_model.deter_policy.select_actions, args.env,00)
                # 设定 args.eval_episodes 为 10，评估10个episodes再取平均
                mean_episode_reward = eval_policy(score_model, args.env_id, args.seed, 10, args.discrete_action, device='cpu', env_args=args.env_args)
                writer.add_scalar({"eval/reward": mean_episode_reward}, epoch+1)

            writer.add_scalar("JAL/v_loss", score_model.q[0].v_loss.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("JAL/q_loss", score_model.q[0].q_loss.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("JAL/q", score_model.q[0].q.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("JAL/v", score_model.q[0].v.detach().cpu().numpy(), epoch+1)
            # policy loss = bc_loss * exp(Q-V)
            writer.add_scalar("JAL/policy_loss", score_model.policy_loss.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("JAL/mean policy loss", avg_critic_loss / num_items, epoch+1)
            writer.add_scalar("JAL/mean bc loss", avg_bc_loss / num_items, epoch+1)
            # policy loss 用了 cosineAnnealing 150w steps
            writer.add_scalar("JAL/lr", score_model.deter_policy_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)

            # 记录各网络的梯度范数
            q_grad_norm = get_grad_norm(score_model.q[0].q0.parameters())
            v_grad_norm = get_grad_norm(score_model.q[0].vf.parameters())
            policy_grad_norm = get_grad_norm(score_model.deter_policy.parameters())
            # 记录每个step的梯度范数
            writer.add_scalar('Check/q_grad_norm', q_grad_norm, epoch+1)
            writer.add_scalar('Check/v_grad_norm', v_grad_norm, epoch+1)
            writer.add_scalar('Check/policy_grad_norm', policy_grad_norm, epoch+1)
            writer.add_scalar('Check/adv', score_model.q[0].adv.detach().cpu().numpy(), epoch+1)
            writer.add_scalar('Check/q_std', score_model.q[0].q_std.detach().cpu().numpy(), epoch+1)
            writer.add_scalar('Check/q_max', score_model.q[0].q_max.detach().cpu().numpy(), epoch+1)
            writer.add_scalar('Check/q_min', score_model.q[0].q_min.detach().cpu().numpy(), epoch+1)
            writer.add_scalar('Check/v_next', score_model.q[0].v_next.detach().cpu().numpy(), epoch+1)
            writer.add_scalar('Check/v_next_std', score_model.q[0].v_next_std.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("Check/v_loss", score_model.q[0].v_loss.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("Check/q_loss", score_model.q[0].q_loss.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("Check/q", score_model.q[0].q.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("Check/v", score_model.q[0].v.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("Check/rewards", score_model.q[0].r.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("Check/mean_dones", score_model.q[0].d.detach().cpu().numpy(), epoch+1)
            writer.add_scalar("Check/q_target", score_model.q[0].target.detach().cpu().numpy(), epoch+1)
            
        
        """ Save models """
        if args.save_model and epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New lowest loss in epoch {}, Save best models".format(epoch))
            if args.mixed_data:
                torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_mix", args.data_type, "JAL", "best_critic.pth"))
            else:
                torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", "JAL", "best_critic.pth"))
                # SRPO_premodels/env_id_level/JAL/best_critic.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            print("Save models: Epoch {}".format(epoch))
            if args.mixed_data:
                torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_mix", args.data_type, "JAL", "critic_epoch{}.pth".format(epoch)))
            else:
                torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", "JAL", "critic_epoch{}.pth".format(epoch)))
            # SRPO_premodels/env_id_level/JAL/critic_150.pth


def critic(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 要添加 2ant 4ant
    if args.env_id in ['HalfCheetah-v2']:
        env_args = {"scenario": args.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
        env = MujocoMulti(env_args=env_args)
        args.env_args = env_args
        env.seed(args.seed + 100)
        env_info = env.get_env_info()
    elif args.env_id == 'bandit':
        env = ContinuousBanditEnv()
        env_info = env.env_info
    elif args.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        env = make_parallel_env(args.env_id, args.seed, args.discrete_action)
        env_args, env_info = None, None
        args.env_args = env_args
    else:   
        print("Env not in MaMujoco, bandit, MPE")



    if args.env_id in ['HalfCheetah-v2', 'bandit']:
        each_state_shape = [env_info['state_shape'] for _ in env.observation_space]
        # MaMujuco use obs as input
        each_obs_shape = [env_info['obs_shape'] for _ in env.observation_space]
        each_action_shape = [acsp.shape[0] for acsp in env.action_space]
        agent_num = len(each_action_shape) 
        state_dim = each_obs_shape[0]
        action_dim = each_action_shape[0]
    elif args.env_id in ['simple_spread']:
        each_state_shape = [obsp.shape[0] for obsp in env.observation_space]
        each_action_shape = [acsp.shape[0] for acsp in env.action_space]
        each_action_max = [acsp.high[0] for acsp in env.action_space]
        agent_num = len(each_action_shape)   # agent 数量
        state_dim = each_state_shape[0]
        action_dim = each_action_shape[0]
        action_max = each_action_max[0]
        args.predator_nums = agent_num
    elif args.env_id in ['simple_tag', 'simple_world']:
        adversary_indices = [i for i, agent_type in enumerate(env.agent_types) if agent_type == 'adversary']
        # 去除 agent 预训练的数据，不需要score model
        each_state_shape = [env.observation_space[i].shape[0] for i in adversary_indices]
        each_action_shape = [env.action_space[i].shape[0] for i in adversary_indices]
        each_action_max = [env.action_space[i].high[0] for i in adversary_indices]
        agent_num = len(adversary_indices)   # 只加载012这三个agent的数据，3号agent是prey，不需要训练和data
        state_dim = each_state_shape[0]
        action_dim = each_action_shape[0]
        action_max = each_action_max[0]
        args.predator_nums = agent_num

        # 加一个args.prey_nums

        # 这里agents区分prey和predators

    
    
    if args.srpo_mode == 'IND':
        score_model= [MASRPO_IQL(input_dim=state_dim+action_dim,
                                 output_dim=action_dim,
                                 args=args).to(args.device) for agent in range(agent_num)]
        for model in score_model:
            model.q[0].to(args.device)
    elif args.srpo_mode == 'JAL' or args.srpo_mode == 'CTDE':
        score_model= MASRPO_IQL(input_dim=(state_dim+action_dim)*agent_num,
                                output_dim=action_dim*agent_num,
                                args=args).to(args.device)
        score_model.q[0].to(args.device)
        # Input is concated obs

    # Load Buffer
    if args.env_id in ['HalfCheetah-v2', 'bandit']:
        replay_buffer = ReplayBuffer(args.buffer_length,
                                     agent_num,
                                     [env_info['obs_shape'] for _ in env.observation_space],
                                     [acsp.shape[0] for acsp in env.action_space],
                                     is_mamujoco=True,
                                     state_dims=[env_info['state_shape'] for _ in env.observation_space],
                                     device = args.device)
    elif args.env_id in ['simple_spread', 'simple_tag', 'simple_world']: 
        replay_buffer = ReplayBuffer(args.buffer_length,
                                     agent_num,
                                     each_state_shape,
                                     [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space],
                                     device = args.device)
    replay_buffer.load_batch_data(args.dataset_dir, rew_scale=args.rew_scale)


    """ Train Log Dir """
    tb_log_path = os.path.join("./logs_SRPO_critic_pretrain", "{}_{}_dataset{}_{}_seed{}_buffer{}_{}".format(str(args.env_id), args.data_type, args.dataset_num ,args.srpo_mode, args.seed, args.buffer_length, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")) )
    writer = SummaryWriter(log_dir=tb_log_path)

    """ Model Saving Dir """
    if not os.path.exists(os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", args.srpo_mode)):
        os.makedirs(os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", args.srpo_mode))
    if args.mixed_data:
        if not os.path.exists(os.path.join("./SRPO_premodels", f"{args.env_id}_mix", args.data_type, args.srpo_mode)):
            os.makedirs(os.path.join("./SRPO_premodels", f"{args.env_id}_mix", args.data_type, args.srpo_mode))


    if args.srpo_mode == 'IND':
        print(f"training IND critics, {agent_num} agents")
        for i in range(agent_num):
            train_ind_critic(args, score_model[i], replay_buffer, i, writer, env_id=args.env_id, start_epoch=0)
    elif args.srpo_mode == 'JAL' or 'CTDE':
        print(f"training JAL critics, {agent_num} agents")
        train_joint_critic(args, score_model, replay_buffer, writer, env_id=args.env_id, start_epoch=0)
    print("finished")



def pretrain_critic_args():
    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='simple_spread', type=str, help="Name of environment")  # HalfCheetah-v2 / bandit 
    parser.add_argument("--data_type", default='medium', type=str)
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")
    # train mode
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=0, type=int, help='cuda number')
    parser.add_argument("--srpo_mode", default='JAL', type=str)   # IND 或者 JAL
    # params for networks
    # parser.add_argument("--actor_blocks", default=2, type=int)   # 在IQL中用不到，只在diffusion中用到scorenet IDQL
    parser.add_argument("--q_layer", default=2, type=int)   # q_layer 是 IQL 中 TwinQ critic 的层数 
    parser.add_argument("--batch_size", default=512, type=int)
    # params for buffer and data
    parser.add_argument('--dataset_dir', default='/data/qiaodan/code/diffmarl/datasets', type=str)
    parser.add_argument("--use_gpu", default=True, type=bool, help='use cuda or not')
    parser.add_argument("--buffer_length", default=int(1e6), type=int)   # omar数据集mamujoco和mpe都是1e6数据量，medium replay会少一些到1e5
    parser.add_argument("--rew_scale", default=1.0, type=float)
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--iql_critic_lr", default=3e-4, type=float)

    # training/eval epochs
    parser.add_argument("--training_epoch", default=200, type=int)
    parser.add_argument("--eval_interval", default=5, type=int)
    parser.add_argument("--save_interval", default=20, type=int)

    # MPE, default False = continuous，如果命令行不指定，则采用默认值，如果指定了，采用True
    parser.add_argument("--discrete_action", action='store_true', default=False)
    # mixed datasets
    parser.add_argument("--mixed_data", action='store_true', default=False)

    config = parser.parse_args()


    # 只用于 mamujoco
    # 需要增加 2ant 4ant
    if config.env_id == "HalfCheetah-v2":      
        config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
    elif config.env_id == "2-ant":
        config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}  # 需要更新确认


    # combine dir
    if config.env_id in ['HalfCheetah-v2', 'simple_spread', 'simple_tag', 'simple_world']:
        if config.mixed_data:
            config.dataset_dir = '/data/qiaodan/code/diffmarl/datasets/mix_hc' + '/' + config.data_type + '/' + 'mixed_data'
        else:
            config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)
    elif config.env_id == 'bandit':
        config.dataset_dir = config.dataset_dir + '/bandit'


    if config.use_gpu:
        config.device = f"cuda:{config.device}"
    else:
        config.device = "cpu"

    return config


if __name__ == "__main__":
    args = pretrain_critic_args()
    critic(args)