""" Pretrain joint Q(s,a) or advantage Q(s, a-i, ai) """

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

from torch.utils.tensorboard import SummaryWriter
from algorithms.SRPO import MASRPO_IQL
# from utils import get_args, pallaral_simple_eval_policy

from utils.buffer import ReplayBuffer
from bandit import ContinuousBanditEnv


# Q(s, a_i)
def train_ind_critic(args, score_model, data_loader, agent_num, writer, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 1
    epoch_save_interval = 20

    best_loss = 1e5

    for epoch in tqdm_epoch:
        avg_critic_loss = 0.
        avg_bc_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size, to_gpu=True)
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
            # if (epoch % 5 == 4) or epoch==0:
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
            torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "IND", "best_critic_{}.pth".format(agent_num)))
            # SRPO_premodels/env_id/IND/best_critic_i.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            print("Save critic models: Epoch {}".format(epoch))
            torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "IND", "critic_{}_epoch{}.pth".format(agent_num, epoch)))
            # SRPO_premodels/env_id_level/IND/critic_1_epoch150.pth

# Q(s, a_0, a_n)
def train_joint_critic(args, score_model, data_loader, writer, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 1
    epoch_save_interval = 20
    best_loss = 1e5

    for epoch in tqdm_epoch:
        avg_critic_loss = 0.
        avg_bc_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size, to_gpu=True)
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
            # 简化训练，不做evaluate IQL
            # if (epoch % 5 == 4) or epoch==0:
            #     mean, std = pallaral_simple_eval_policy(score_model.deter_policy.select_actions,args.env,00)
            #     args.run.log({"eval/rew{}".format("deter"): mean}, step=epoch+1)

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
        
        """ Save models """
        if args.save_model and epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New lowest loss in epoch {}, Save best models".format(epoch))
            torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "JAL", "best_critic.pth"))
                # SRPO_premodels/env_id_level/JAL/best_critic.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            print("Save models: Epoch {}".format(epoch))
            torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "JAL", "critic_epoch{}.pth".format(epoch)))
            # SRPO_premodels/env_id_level/JAL/critic_150.pth





# A(s, a_i-, a_i)
# def train_adv_critic(args, score_model, data_loader, start_epoch=0):
#     n_epochs = 150
#     tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
#     # evaluation_inerval = 4
#     evaluation_inerval = 1
#     save_interval = 10

#     for epoch in tqdm_epoch:
#         avg_loss = 0.
#         num_items = 0
#         for _ in range(10000):
#             data = data_loader.sample(256)
#             loss2 = score_model.update_iql(data)
#             avg_loss += 0.0
#             num_items += 1
#         tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
#         if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
#             if (epoch % 5 == 4) or epoch==0:
#                 mean, std = pallaral_simple_eval_policy(score_model.deter_policy.select_actions,args.env,00)
#                 args.run.log({"eval/rew{}".format("deter"): mean}, step=epoch+1)
#             args.run.log({"loss/v_loss": score_model.q[0].v_loss.detach().cpu().numpy()}, step=epoch+1)
#             args.run.log({"loss/q_loss": score_model.q[0].q_loss.detach().cpu().numpy()}, step=epoch+1)
#             args.run.log({"loss/q": score_model.q[0].q.detach().cpu().numpy()}, step=epoch+1)
#             args.run.log({"loss/v": score_model.q[0].v.detach().cpu().numpy()}, step=epoch+1)
#             args.run.log({"loss/policy_loss": score_model.policy_loss.detach().cpu().numpy()}, step=epoch+1)
#             args.run.log({"info/lr": score_model.deter_policy_optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch+1)
#         if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
#             torch.save(score_model.q[0].state_dict(), os.path.join("./SRPO_model_factory", str(args.expid), "critic_ckpt{}.pth".format(epoch+1)))



def critic(args):
    for dir in ["./SRPO_model_factory"]:
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent_num = len(each_action_shape) 
    state_dim = each_obs_shape[0]
    action_dim = each_action_shape[0]

    if args.srpo_mode == 'IND':
        score_model= [MASRPO_IQL(input_dim=state_dim+action_dim, output_dim=action_dim, args=args).to(args.device) for agent in range(agent_num)]
        for model in score_model:
            model.q[0].to(args.device)
    elif args.srpo_mode == 'JAL' or args.srpo_mode == 'CTDE':
        score_model= MASRPO_IQL(input_dim=(state_dim+action_dim)*agent_num, output_dim=action_dim*agent_num, args=args).to(args.device)
        score_model.q[0].to(args.device)
        # In MaMujuco, input is concated obs


    replay_buffer = ReplayBuffer(
            args.buffer_length, agent_num,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space], device = args.device
        )
    replay_buffer.load_batch_data(args.dataset_dir, rew_scale = args.rew_scale)

    """ Train Log Dir """
    tb_log_path = os.path.join("./logs_SRPO_critic_pretrain", "{}_{}_{}_seed{}_buffer{}_{}".format(str(args.env_id), args.data_type, args.srpo_mode, args.seed, args.buffer_length, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")) )
    writer = SummaryWriter(log_dir=tb_log_path)

    """ Model Saving Dir """
    if not os.path.exists(os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", args.srpo_mode)):
        os.makedirs(os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", args.srpo_mode))

    print("training critic")
    if args.srpo_mode == 'IND':
        for i in range(agent_num):
            train_ind_critic(args, score_model[i], replay_buffer, i, writer, start_epoch=0)
    elif args.srpo_mode == 'JAL' or 'CTDE':
        train_joint_critic(args, score_model, replay_buffer, writer, start_epoch=0)
    print("finished")


def pretrain_critic_args():
    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='HalfCheetah-v2', type=str, help="Name of environment")
    parser.add_argument("--data_type", default='expert', type=str)
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")
    # train mode
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_gpu", default=True, type=bool, help='use cuda or not')
    parser.add_argument("--device", default=1, type=int, help='cuda number')
    parser.add_argument("--srpo_mode", default='IND', type=str)
    # params for networks
    parser.add_argument("--actor_blocks", default=3, type=int)
    parser.add_argument("--q_layer", default=2, type=int)
    parser.add_argument("--batch_size", default=512, type=int)


    parser.add_argument('--dataset_dir', default='/home/qiaodan/Code/diffmarl/datasets', type=str)

    # params for buffer and data
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--rew_scale", default=1.0, type=float)
    parser.add_argument("--save_model", default=True, type=bool)

    config = parser.parse_args()

    config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}

    # combine dir
    if config.env_id == 'HalfCheetah-v2':
        config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)
    else:
        config.dataset_dir = config.dataset_dir + '/' + config.env_id
    # config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)

    if config.use_gpu:
        config.device = f"cuda:{config.device}"
    else:
        config.device = "cpu"

    return config


if __name__ == "__main__":
    args = pretrain_critic_args()
    critic(args)