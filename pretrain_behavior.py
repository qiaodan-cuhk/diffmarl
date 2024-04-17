""" Pretrain Joint Policy Diffusion and Individual Diffusion to get scores"""

import functools
import os

import numpy as np
import torch
import tqdm
import argparse
import datetime

try:
    from multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('MujocoMulti not installed')

from algorithms.SRPO import MASRPO_Behavior
from torch.utils.tensorboard import SummaryWriter

# from utils import get_args, marginal_prob_std

def marginal_prob_std(t, device="cuda",beta_1=20.0,beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$."""    
    # t = torch.tensor(t, device=device)
    t = t.clone().detach().to(device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std


from utils.buffer import ReplayBuffer

def train_ind_behavior(args, score_model, data_loader, agent_num, start_epoch=0):
    n_epochs = 20
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 1
    save_interval = 1

    best_loss = 1e3

    tb_log_path = os.path.join("./logs_SRPO_diffusion_pretrain", "{}_{}_seed{}_{}".format(str(args.env_id), "IND", args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")) )
    writer = SummaryWriter(log_dir=tb_log_path)

    """ Model saving dir """
    if not os.path.exists(os.path.join("./SRPO_premodels", str(args.env_id), "IND")):
        os.makedirs(os.path.join("./SRPO_premodels", str(args.env_id), "IND"))

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size)
            data_i = data[agent_num]
            loss2 = score_model.update_behavior(data_i)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
            writer.add_scalar('loss in episode | agent {}'.format(agent_num), loss2, step_in_epoch)
        tqdm_epoch.set_description('Average Loss of Agent {}: {:5f}'.format(agent_num, avg_loss / num_items))
        
        epoch_loss = score_model.loss.detach().cpu().numpy()

        """ Log by tensorboard"""
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            writer.add_scalar('loss of diffusion model agent {}'.format(agent_num), epoch_loss, epoch+1)
            # args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)


        """ Save models """
        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                print("New lowest loss in epoch {}, Save models".format(epoch))
                torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", str(args.env_id), "IND", "diffusion_{}.pth".format(agent_num)))
                # SRPO_premodels/env_id/IND/diffusion_i.pth
        


def train_joint_behavior(args, score_model, data_loader, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    # evaluation_inerval = 4
    evaluation_inerval = 1
    save_interval = 20

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = data_loader.sample(args.batch_size)
            # 怎么把 dict = {state} 与 score update 里的 s 挂钩
            loss2 = score_model.update_behavior(data)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", str(args.env_id), "JAL", "diffusion.pth"))
            # SRPO_premodels/env_id/JAL/diffusion.pth


def behavior(args):
    for dir in ["./SRPO_premodels"]:
        if not os.path.exists(dir):
            os.makedirs(dir)


    if args.env_id in ['HalfCheetah-v2']:
        env = MujocoMulti(env_args=args.env_args)
        env.seed(args.seed + 100)
        env_info = env.get_env_info()




    # env.action_space.seed(args.seed)

    each_state_shape = [env_info['state_shape'] for _ in env.observation_space]
    # MaMujuco use obs as input
    each_obs_shape = [env_info['obs_shape'] for _ in env.observation_space]
    each_action_shape = [acsp.shape[0] for acsp in env.action_space]
    
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent_num = len(each_action_shape) 

    state_dim = each_obs_shape[0]
    action_dim = each_action_shape[0]

    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=20.0)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    if args.srpo_mode == 'IND' or 'CTDE':
        score_model= [MASRPO_Behavior(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device) for agent in range(agent_num)]
    elif args.srpo_mode == 'JAL':
        score_model= MASRPO_Behavior(input_dim=state_dim+action_dim*agent_num, output_dim=action_dim*agent_num, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)


    replay_buffer = ReplayBuffer(
            args.buffer_length, agent_num,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space], device = args.device
        )
    replay_buffer.load_batch_data(args.dataset_dir, rew_scale = args.rew_scale)

    print("training behavior")
    if args.srpo_mode == 'CTDE' or 'IND':
        for i in range(agent_num):
            train_ind_behavior(args, score_model[i], replay_buffer, i, start_epoch=0)
    elif args.srpo_mode == 'JAL':
        train_joint_behavior(args, score_model, replay_buffer, start_epoch=0)
    print("finished")

def pretrain_behavior_args():

    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='HalfCheetah-v2', type=str, help="Name of environment")
    parser.add_argument("--data_type", default='expert', type=str)
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_gpu", default=True, type=bool, help='use cuda or not')
    parser.add_argument("--device", default=1, type=int, help='cuda number')

    parser.add_argument('--dataset_dir', default='/home/qiaodan/Code/diffmarl/datasets', type=str)

    parser.add_argument("--srpo_mode", default='IND', type=str)
    parser.add_argument("--actor_blocks", default=2, type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    # params for buffer and data
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--rew_scale", default=1.0, type=float)
    parser.add_argument("--save_model", default=True, type=bool)


    config = parser.parse_args()

    config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}

    # combine dir
    config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)

    if config.use_gpu:
        config.device = f"cuda:{config.device}"
    else:
        config.device = "cpu"

    return config


if __name__ == "__main__":
    args = pretrain_behavior_args()
    behavior(args)
