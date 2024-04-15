""" Pretrain Joint Policy Diffusion and Individual Diffusion to get scores"""

import functools
import os

import gym
import numpy as np
import torch
import tqdm
import argparse

from algorithms.SRPO import SRPO_Behavior
# from utils import get_args, marginal_prob_std

def marginal_prob_std(t, device="cuda",beta_1=20.0,beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """    
    # t = torch.tensor(t, device=device)
    t = t.clone().detach().to(device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std



def train_ind_behavior(args, score_model, data_loader, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    # evaluation_inerval = 4
    evaluation_inerval = 1
    save_interval = 20

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = data_loader.sample(2048)
            loss2 = score_model.update_behavior(data)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", str(args.expid), args.srpo_mode, "diffusion_{}.pth".format(index)))
            # SRPO_premodels/expid/IND/diffusion_i.pth

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
            data = data_loader.sample(2048)
            loss2 = score_model.update_behavior(data)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", str(args.expid), args.srpo_type, "diffusion.pth".format(epoch+1)))
            # SRPO_premodels/expid/JAL/diffusion.pth


def behavior(args):
    for dir in ["./SRPO_premodels"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./SRPO_premodels", str(args.expid))):
        os.makedirs(os.path.join("./SRPO_premodels", str(args.expid)))

    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=20.0)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= SRPO_Behavior(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)

    dataset = 

    print("training behavior")
    if args.diffusion_mode == 'CTDE' or 'IND':
        train_ind_behavior(args, score_model, dataset, start_epoch=0)
    elif args.diffusion_mode == 'JAL':
        train_joint_behavior(args, score_model, dataset, start_epoch=0)
    print("finished")

def pretrain_behavior_args():

    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='simple_spread', type=str, help="Name of environment")
    parser.add_argument("--data_type", default='expert', type=str)
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")


    parser.add_argument("--srpo_mode", default='IND', type=str)

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    args = pretrain_behavior_args()
    behavior(args)
