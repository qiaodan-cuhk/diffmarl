""" Pretrain Joint Policy Diffusion and Individual Diffusion to get scores"""

import functools
import os
import numpy as np
import torch
import tqdm
import argparse
import datetime

from utils.buffer import ReplayBuffer

try:
    from multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('MujocoMulti not installed')

from bandit import ContinuousBanditEnv

from algorithms.SRPO import MASRPO_Behavior
from torch.utils.tensorboard import SummaryWriter


def marginal_prob_std(t, device="cuda",beta_1=20.0,beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$."""    
    # t = torch.tensor(t, device=device)
    t = t.clone().detach().to(device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std


def train_ind_behavior(args, score_model, data_loader, agent_num, writer, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 1
    epoch_save_interval = 50
    best_loss = 1e3

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size)
            data_i = data[agent_num]
            loss2 = score_model.update_behavior(data_i)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
            writer.add_scalar('agent {}/episode loss'.format(agent_num), loss2, step_in_epoch)
        tqdm_epoch.set_description('Average Loss of Agent {}: {:5f}'.format(agent_num, avg_loss / num_items))

        epoch_loss = score_model.loss.detach().cpu().numpy()
        writer.add_scalar("agent {}/lr".format(agent_num), score_model.diffusion_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)

        """ Log by tensorboard"""
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            writer.add_scalar('agent {}/epoch loss'.format(agent_num), epoch_loss, epoch+1)
            writer.add_scalar('agent {}/mean epoch loss'.format(agent_num), avg_loss / num_items, epoch+1)
            # args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

        """ Save models """
        if args.save_model and epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New lowest loss in epoch {}, Save best models".format(epoch))
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "IND", "best_diffusion_{}.pth".format(agent_num)))
            # SRPO_premodels/env_id_level/IND/best_diffusion_i.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            print("Save models: Epoch {}".format(epoch))
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "IND", "diffusion_{}_epoch{}.pth".format(agent_num, epoch)))
            # SRPO_premodels/env_id_level/IND/diffusion_i_epoch150.pth   
        

def train_joint_behavior(args, score_model, data_loader, writer, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 1
    epoch_save_interval = 50
    best_loss = 1e3

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size)

            # we need to concate marl datasets with [{}, {}] into one dict
            d0 = data[0]
            d1 = data[1]
            data_concate = {}
            for item in ["obs", "action", "next_obs", "next_action"]:
                data_concate[item] = torch.cat((d0[item], d1[item]), dim=1).to(args.device)
                assert data_concate[item].size()[0] == args.batch_size
            data_concate["state"] = d0["state"].to(args.device)
            data_concate["next_state"] = d0["next_state"].to(args.device)
            data_concate["rewards"] = d0["rewards"].to(args.device)
            data_concate["done"] = d0["done"].to(args.device)

            loss2 = score_model.update_behavior(data_concate)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
            writer.add_scalar('JAL/episode loss', loss2, step_in_epoch)
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        epoch_loss = score_model.loss.detach().cpu().numpy()
        writer.add_scalar("JAL/lr", score_model.diffusion_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            writer.add_scalar('JAL/epoch loss', epoch_loss, epoch+1)
            writer.add_scalar('JAL/mean epoch loss', avg_loss / num_items, epoch+1)

        """ Save models """
        if args.save_model and epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New lowest loss in epoch {}, Save best models".format(epoch))
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "JAL", "best_diffusion.pth"))
                # SRPO_premodels/env_id_level/JAL/best_diffusion.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            print("Save models: Epoch {}".format(epoch))
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "JAL", "diffusion_{}.pth".format(epoch)))
            # SRPO_premodels/env_id_level/JAL/diffusion_150.pth


def behavior(args):
    for dir in ["./SRPO_premodels"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if args.env_id in ['HalfCheetah-v2']:
        env = MujocoMulti(env_args=args.env_args)
        env.seed(args.seed + 100)
        env_info = env.get_env_info()
    elif args.env_id == 'bandit':
        env = ContinuousBanditEnv()
        env_info = env.env_info

    each_state_shape = [env_info['state_shape'] for _ in env.observation_space]
    # MaMujuco use obs as input
    each_obs_shape = [env_info['obs_shape'] for _ in env.observation_space]
    each_action_shape = [acsp.shape[0] for acsp in env.action_space]
    agent_num = len(each_action_shape) 
    state_dim = each_obs_shape[0]
    action_dim = each_action_shape[0]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=20.0)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    if args.srpo_mode == 'IND' or args.srpo_mode == 'CTDE':
        score_model= [MASRPO_Behavior(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device) for agent in range(agent_num)]
    elif args.srpo_mode == 'JAL':
        score_model= MASRPO_Behavior(input_dim=(state_dim+action_dim)*agent_num, output_dim=action_dim*agent_num, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
        # JAL model is p(all_a|all_obs)

    if args.env_id in ['HalfCheetah-v2', 'bandit']:
        replay_buffer = ReplayBuffer(
                args.buffer_length, agent_num,
                [env_info['obs_shape'] for _ in env.observation_space],
                [acsp.shape[0] for acsp in env.action_space],
                is_mamujoco=True,
                state_dims=[env_info['state_shape'] for _ in env.observation_space], device = args.device
            )
        replay_buffer.load_batch_data(args.dataset_dir, rew_scale = args.rew_scale)

    """ Train Log Dir """
    tb_log_path = os.path.join("./logs_SRPO_diffusion_pretrain", "{}_{}_{}_seed{}_buffer{}_{}".format(str(args.env_id), args.data_type, args.srpo_mode, args.seed, args.buffer_length, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")) )
    writer = SummaryWriter(log_dir=tb_log_path)

    """ Model Saving Dir """
    if not os.path.exists(os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", args.srpo_mode)):
        os.makedirs(os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", args.srpo_mode))

    print("training behavior")
    if args.srpo_mode == 'CTDE' or args.srpo_mode == 'IND':
        for i in range(agent_num):
            train_ind_behavior(args, score_model[i], replay_buffer, i, writer, start_epoch=0)
    elif args.srpo_mode == 'JAL':
        train_joint_behavior(args, score_model, replay_buffer, writer, start_epoch=0)
    print("finished")

def pretrain_behavior_args():
    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='bandit', type=str, help="Name of environment") # HalfCheetah-v2 bandit  HalfCheetah-v2
    parser.add_argument("--data_type", default='expert', type=str)  # medium-replay
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")
    # train mode
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=1, type=int, help='cuda number')
    parser.add_argument("--srpo_mode", default='IND', type=str)
    # params for Score Network
    parser.add_argument("--actor_blocks", default=2, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--t_GassProj_dims", default=32, type=int)
    parser.add_argument("--t_embed_dims", default=64, type=int)
    parser.add_argument("--sa_embed_dims", default=32, type=int)
    parser.add_argument("--resnet_hidden_dim", default=512, type=int)   # ResNet MLP hidden dim 
    parser.add_argument("--learning_rates", default=3e-4, type=float)
    parser.add_argument("--lr_anneal", default=True, type=bool)


    """   UnChangable params by users   """
    parser.add_argument('--dataset_dir', default='/home/qiaodan/Code/diffmarl/datasets', type=str)
    parser.add_argument("--use_gpu", default=True, type=bool, help='use cuda or not')
    # params for buffer and data
    parser.add_argument("--buffer_length", default=int(1e6), type=int)  # full marl dataset is 1e6 
    parser.add_argument("--rew_scale", default=1.0, type=float)
    parser.add_argument("--save_model", default=True, type=bool)

    # parser.add_argument("--eval_models", default=False, type=bool)

    config = parser.parse_args()

    config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
    # combine dir
    if config.env_id == 'HalfCheetah-v2':
        config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)
    else:
        config.dataset_dir = config.dataset_dir + '/' + config.env_id

    if config.use_gpu:
        config.device = f"cuda:{config.device}"
    else:
        config.device = "cpu"

    return config


if __name__ == "__main__":
    args = pretrain_behavior_args()
    behavior(args)
