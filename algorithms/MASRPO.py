import torch
import torch.nn.functional as F

from utils.misc import soft_update, average_gradients
from utils.agents import DDPGAgent
import numpy as np
from utils.noise import action_noise

import copy
import torch.nn as nn
from .SRPO import SRPO

from tensorboard_logger import log_value

import functools

# SRPO marginal_prob_std\\\
def marginal_prob_std(t, device="cuda",beta_1=20.0,beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """    
    # t = torch.tensor(t, device=device)
    t = t.clone().detach().to(device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std

"""
Joint action learning: central critic, joint policy diffusion
Independent learning: independent critic, indpendent policy diffusion
QMIX learning: central critic -> VD ind critic, ind policy diffusion
Seq: central critic, joint policy diffusion, partial diffusion to regularize ind policy
"""

# MARL SRPO Algos
class IND_SRPO(object):
    def __init__(
        self, 
        agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=20,
        device = 'cpu',
        adv_init_params=None,
        gamma=0.99, # RL discount gamma
        tau=0.01,  # 这个 tau 是用来 target network soft update的
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        env_id=None,
        batch_size = 100,
        config = None,
        **kwargs
    ):
        self.env_id = env_id
        self.is_mamujoco = True if self.env_id == 'HalfCheetah-v2' else False

        assert (ma == agent_max_actions[0] for ma in agent_max_actions)
        self.max_action = agent_max_actions[0]
        self.min_action = -self.max_action
        self.hidden_dim = hidden_dim  # 64 for DDPG

        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.tau = tau
  
        self.agent_init_params = agent_init_params
        self.state_dim = self.agent_init_params[0]['state_dim']
        self.action_dim = self.agent_init_params[0]['action_dim']
        self.device = device
        
        self.gamma = gamma
        self.lr = lr
        self.discrete_action = discrete_action

        self.T = denoise_steps
        self.batch_size = batch_size

        for k, v in kwargs.items():
            setattr(self, k, v)

        marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device, beta_1=20.0)

        self.agents = [SRPO(input_dim = self.state_dim+self.action_dim, output_dim=self.action_dim, marginal_prob_std=marginal_prob_std_fn, args=config)]
        for age in self.agents:
            age.q[0].to(self.device)


        if self.env_id in ['simple_tag', 'simple_world']:
            self.num_predators = len(agent_init_params)
            self.num_preys = len(adv_init_params)

            self.preys = [DDPGAgent(lr=lr, discrete_action=self.discrete_action, hidden_dim=self.hidden_dim, **params) for params in adv_init_params]

        self.niter = 0
  

    # @property
    # def policies(self):
    #     return [a.actor for a in self.agents]
    # # self.policy = Diffusion()
    # # action = self.policy.forward(obs)

    # @property
    # def target_policies(self):
    #     return [a.ema_model for a in self.agents]
    # # self.ema_model = copy.deepcopy(self.actor)
    # # 但是这个并不用作 target policy，没有 soft update，soft只作用于 diffusion critic


    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        actions = []

        # nagents = agents + prey (if have)
        for i, obs in zip(range(self.nagents), observations):   
            if self.env_id in ['simple_world', 'simple_tag']:
                if i < self.num_predators:
                    predator_action = self.agents[i].SRPO_policy.select_actions(obs)  # SRPO use Dilac sample action
                    actions.append(predator_action)
                else:
                    prey_action = self.preys[i - self.num_predators].step(obs, explore=False)
                    actions.append(prey_action)
            else:
                action = self.agents[i].SRPO_policy.select_actions(obs)
                actions.append(action)
        return actions

        # actions = [array([0.99983406, 0...e=float32), array([0.9950481 , 0...e=float32), array([ 0.44896033, ...e=float32)]
        # observations = [tensor([[ 0.0000,  0... 0.0000]]), tensor([[ 0.0000,  0... 0.0000]]), tensor([[ 0.0000,  0... 0.0000]])]
        # obs = tensor([1, 18])
    

    def update(self, sample, agent_i, t):

        curr_agent = self.agents[agent_i]

        # data reconstruction
        # 参考OMAR，mamujoco提供了state，但是训练用的还是obs
        if self.is_mamujoco:
            sample_bridge = {"s": sample["obs"],
                             "a": sample["action"],
                             "r": sample["rewards"],
                             "s_": sample["next_obs"],
                             "d": sample["done"]
            }
        else:
            sample_bridge = {"s": sample["obs"],
                             "a": sample["action"],
                             "r": sample["rewards"],
                             "s_": sample["next_obs"],
                             "d": sample["done"]
            }

        loss_tot, epsilon, guidance = curr_agent.update_SRPO_policy(sample_bridge)
        
        """ logging metric """
        if t % self.logging_interval == 0 and not self.no_log:
            dic = {}
            dic.update({"SRPO loss"+str(agent_i): loss_tot})
            dic.update({"diffusion loss"+str(agent_i): epsilon})
            dic.update({"Q gradient"+str(agent_i): guidance})
            
            log_and_print(list(dic.keys()), list(dic.values()), t, multi=True)

    # prepare train() or eval() 
    def prep_training(self, device='cpu'):

        for a in self.agents:
            a.diffusion_behavior.train()
            a.SRPO_policy.train()
            a.q[0].train()

        fn = lambda x: x.to(device)   
        for a in self.agents:
            a.diffusion_behavior = fn(a.diffusion_behavior)
            a.SRPO_policy = fn(a.SRPO_policy)
            a.q[0] = fn(a.q[0])

        if self.env_id in ['simple_tag', 'simple_world']:
            for p in self.preys:
                p.policy = fn(p.policy)
                p.target_policy = fn(p.target_policy)

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.diffusion_behavior.eval()
            a.SRPO_policy.eval()
            a.q[0].eval()

        fn = lambda x: x.to(device)

        for a in self.agents:
            a.diffusion_behavior = fn(a.diffusion_behavior)
            a.SRPO_policy = fn(a.SRPO_policy)

        if self.env_id in ['simple_tag', 'simple_world']:
            for p in self.preys:
                p.policy = fn(p.policy)


    @classmethod
    def init_from_env(cls, env, env_id, env_info=None, agent_alg="diffusion", adversary_alg="ddpg",
                       gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                       batch_size=None, denoise_steps=20, config=None, **kwargs):
        """
        Instantiate instance of this class from multi-agent environment
        """

        # create n trainable agents without prey, alg_types = ['diff', 'diff', 'diff']
        if env_id in ['simple_tag', 'simple_world']:
            alg_types = [agent_alg for atype in env.agent_types if atype == 'adversary']
        elif env_id in ['simple_spread']:
            alg_types = [agent_alg for atype in env.agent_types]
        elif env_id in ['HalfCheetah-v2']:
            alg_types = [agent_alg for atype in range(env_info['n_agents'])]

        agent_init_params = []
        all_n_actions = []
        agent_max_actions = []
        adv_init_params = []

        # make agent_init_params, adv_init_params, agent_max_actions, all_n_actions
        if env_id == 'HalfCheetah-v2':
            for agent_idx in range(len(alg_types)):
                acsp = env_info['action_spaces'][agent_idx]
                num_in_pol = env_info['obs_shape']
                num_out_pol = acsp.shape[0]

                agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                
                agent_max_actions.append(acsp.high[0])
                all_n_actions.append(acsp.shape[0])
        else:
            for acsp, obsp, agent_type in zip(env.action_space, env.observation_space, env.agent_types):
                num_in_pol = obsp.shape[0]
                num_out_pol = acsp.shape[0]
                num_in_critic = num_in_pol + num_out_pol

                if env_id in ['simple_spread']:
                    agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                    agent_max_actions.append(acsp.high[0])
                else:
                    if agent_type == 'adversary':  # adversary 是猎人，agent 是猎物
                        agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                        agent_max_actions.append(acsp.high[0])
                    elif agent_type == 'agent':
                        adv_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})

                all_n_actions.append(acsp.shape[0])

            for i in range(1, len(all_n_actions)):
                assert (all_n_actions[i] == all_n_actions[0])  # 同时包括了 predator 和 prey 的动作维度


        init_dict = {
            'agent_init_params': agent_init_params,
            'agent_max_actions': agent_max_actions,
            'alg_types': alg_types,
            'denoise_steps': denoise_steps,
            'device': 'cpu',
            'adv_init_params': adv_init_params,
            'gamma': gamma, 
            'tau': tau,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'discrete_action': False,
            'env_id': env_id,
            'batch_size': batch_size,   
            'config': config,
        }

        # if have additional params in kwargs
        init_dict.update(kwargs)
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance


    def load_pretrained_preys(self, filename):
        if not torch.cuda.is_available():
            save_dict = torch.load(filename, map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(filename)

        if self.env_id in ['simple_tag', 'simple_world']:
            prey_params = save_dict['agent_params'][self.num_predators:]

        for i, params in zip(range(self.num_preys), prey_params):
            self.preys[i].load_params_without_optims(params)

        for p in self.preys:
            p.policy.eval()
            p.target_policy.eval()
    


class JAL_SRPO(object):
    def __init__(
        self, 
        agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=20,
        device = 'cpu',
        adv_init_params=None,
        gamma=0.99, # RL discount gamma
        tau=0.01,  # 这个 tau 是用来 target network soft update的
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        env_id=None,
        batch_size = 100,
        config = None,
        **kwargs
    ):
        self.env_id = env_id
        self.is_mamujoco = True if self.env_id == 'HalfCheetah-v2' else False

        assert (ma == agent_max_actions[0] for ma in agent_max_actions)
        self.max_action = agent_max_actions[0]
        self.min_action = -self.max_action
        self.hidden_dim = hidden_dim  # 64 for DDPG

        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.tau = tau
  
        self.agent_init_params = agent_init_params
        # in JAL, the state and action dim are multiplex by agent numbers
        self.state_dim = self.agent_init_params[0]['state_dim'] * self.nagents  # use concate obs as input
        self.action_dim = self.agent_init_params[0]['action_dim'] * self.nagents
        self.device = device
        
        self.gamma = gamma
        self.lr = lr
        self.discrete_action = discrete_action

        self.T = denoise_steps
        self.batch_size = batch_size

        for k, v in kwargs.items():
            setattr(self, k, v)

        marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device, beta_1=20.0)

        self.agents = [SRPO(input_dim = self.state_dim+self.action_dim, output_dim=self.action_dim, marginal_prob_std=marginal_prob_std_fn, args=config)]
        self.agents.q[0].to(self.device)


        if self.env_id in ['simple_tag', 'simple_world']:
            self.num_predators = len(agent_init_params)
            self.num_preys = len(adv_init_params)

            self.preys = [DDPGAgent(lr=lr, discrete_action=self.discrete_action, hidden_dim=self.hidden_dim, **params) for params in adv_init_params]

        self.niter = 0
  

    # @property
    # def policies(self):
    #     return [a.actor for a in self.agents]
    # # self.policy = Diffusion()
    # # action = self.policy.forward(obs)

    # @property
    # def target_policies(self):
    #     return [a.ema_model for a in self.agents]
    # # self.ema_model = copy.deepcopy(self.actor)
    # # 但是这个并不用作 target policy，没有 soft update，soft只作用于 diffusion critic


    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """

        # input torch_obs is [torch(1,18), torch(1,18), torch(1,18)]
        # output actions is [array(2), array(2), array(2)]
        # for JAL, one diff model generate all agents actions

        if self.env_id in ['simple_world', 'simple_tag']:
            obs_predator = observations[:self.num_predators]
            obs_predator = torch.cat(obs_predator, dim=1)
            obs_prey = observations[self.num_predators:]
            
            self.agents[i].SRPO_policy.select_actions(obs)
            predator_actions = self.agents[0].SRPO_policy.select_actions(obs_predator) # input tensor[1,obs_dim * n] output array[1, act_dim * n]
            actions = np.split(predator_actions, self.num_predators, axis=0)

            for i, obs in zip(obs_prey):
                prey_action = self.preys[i].step(obs, explore=False)
                actions.append(prey_action)
        else:
            observations = torch.cat(observations, dim=1)
            joint_actions = self.agents[0].SRPO_policy.select_actions(observations)
            actions = np.split(joint_actions, self.nagents, axis=0)

        assert len(actions[0]) == self.agent_init_params[0]['action_dim']

        return actions

        # actions = [array([0.99983406, 0...e=float32), array([0.9950481 , 0...e=float32), array([ 0.44896033, ...e=float32)]
        # observations = [tensor([[ 0.0000,  0... 0.0000]]), tensor([[ 0.0000,  0... 0.0000]]), tensor([[ 0.0000,  0... 0.0000]])]
        # obs = tensor([1, 18])
    

    def update(self, sample, t):
        
        JAL_agent = self.agents[0]

        jal_obs = [sample_i['obs'] for sample_i in sample]
        jal_obs = torch.cat(jal_obs, dim=1)
        assert jal_obs.size()[0] == self.batch_size

        jal_act = [sample_i['action'] for sample_i in sample]  # nagents * batch_size * act_dim
        jal_act = torch.cat(jal_act, dim=1) # batch_size * (nagents*act_dim)

        jal_rew = sample[0]['rewards']  # [batch_size]
        jal_done = sample[0]['done'] # [batch_size]

        jal_next_obs = [sample_i['next_obs'] for sample_i in sample]
        jal_next_obs = torch.cat(jal_next_obs, dim=1)

        # data reconstruction
        # 参考OMAR，mamujoco提供了state，但是训练用的还是obs
        if self.is_mamujoco:
            sample_bridge = {"s": jal_obs,
                             "a": jal_act,
                             "r": jal_rew,
                             "s_": jal_next_obs,
                             "d": jal_done
            }
        else:
            sample_bridge = {"s": jal_obs,
                             "a": jal_act,
                             "r": jal_rew,
                             "s_": jal_next_obs,
                             "d": jal_done
            }

        loss_tot, epsilon, guidance = JAL_agent.update_SRPO_policy(sample_bridge)
        
        """ logging metric """
        if t % self.logging_interval == 0 and not self.no_log:
            dic = {}
            dic.update({"SRPO loss of JAL": loss_tot})
            dic.update({"diffusion loss of JAL": epsilon})
            dic.update({"Q gradient of JAL": guidance})
            
            log_and_print(list(dic.keys()), list(dic.values()), t, multi=True)

    # prepare train() or eval() 
    def prep_training(self, device='cpu'):

        for a in self.agents:
            a.diffusion_behavior.train()
            a.SRPO_policy.train()
            a.q[0].train()

        fn = lambda x: x.to(device)   
        for a in self.agents:
            a.diffusion_behavior = fn(a.diffusion_behavior)
            a.SRPO_policy = fn(a.SRPO_policy)
            a.q[0] = fn(a.q[0])

        if self.env_id in ['simple_tag', 'simple_world']:
            for p in self.preys:
                p.policy = fn(p.policy)
                p.target_policy = fn(p.target_policy)

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.diffusion_behavior.eval()
            a.SRPO_policy.eval()
            a.q[0].eval()

        fn = lambda x: x.to(device)

        for a in self.agents:
            a.diffusion_behavior = fn(a.diffusion_behavior)
            a.SRPO_policy = fn(a.SRPO_policy)

        if self.env_id in ['simple_tag', 'simple_world']:
            for p in self.preys:
                p.policy = fn(p.policy)


    @classmethod
    def init_from_env(cls, env, env_id, env_info=None, agent_alg="diffusion", adversary_alg="ddpg",
                       gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                       batch_size=None, denoise_steps=20, config=None, **kwargs):
        """
        Instantiate instance of this class from multi-agent environment
        """

        # create n trainable agents without prey, alg_types = ['diff', 'diff', 'diff']
        if env_id in ['simple_tag', 'simple_world']:
            alg_types = [agent_alg for atype in env.agent_types if atype == 'adversary']
        elif env_id in ['simple_spread']:
            alg_types = [agent_alg for atype in env.agent_types]
        elif env_id in ['HalfCheetah-v2']:
            alg_types = [agent_alg for atype in range(env_info['n_agents'])]

        agent_init_params = []
        all_n_actions = []
        agent_max_actions = []
        adv_init_params = []

        # make agent_init_params, adv_init_params, agent_max_actions, all_n_actions
        if env_id == 'HalfCheetah-v2':
            for agent_idx in range(len(alg_types)):
                acsp = env_info['action_spaces'][agent_idx]
                num_in_pol = env_info['obs_shape']
                num_out_pol = acsp.shape[0]

                agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                
                agent_max_actions.append(acsp.high[0])
                all_n_actions.append(acsp.shape[0])
        else:
            for acsp, obsp, agent_type in zip(env.action_space, env.observation_space, env.agent_types):
                num_in_pol = obsp.shape[0]
                num_out_pol = acsp.shape[0]
                num_in_critic = num_in_pol + num_out_pol

                if env_id in ['simple_spread']:
                    agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                    agent_max_actions.append(acsp.high[0])
                else:
                    if agent_type == 'adversary':  # adversary 是猎人，agent 是猎物
                        agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                        agent_max_actions.append(acsp.high[0])
                    elif agent_type == 'agent':
                        adv_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})

                all_n_actions.append(acsp.shape[0])

            for i in range(1, len(all_n_actions)):
                assert (all_n_actions[i] == all_n_actions[0])  # 同时包括了 predator 和 prey 的动作维度


        init_dict = {
            'agent_init_params': agent_init_params,
            'agent_max_actions': agent_max_actions,
            'alg_types': alg_types,
            'denoise_steps': denoise_steps,
            'device': 'cpu',
            'adv_init_params': adv_init_params,
            'gamma': gamma, 
            'tau': tau,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'discrete_action': False,
            'env_id': env_id,
            'batch_size': batch_size,   
            'config': config,
        }

        # if have additional params in kwargs
        init_dict.update(kwargs)
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance


    def load_pretrained_preys(self, filename):
        if not torch.cuda.is_available():
            save_dict = torch.load(filename, map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(filename)

        if self.env_id in ['simple_tag', 'simple_world']:
            prey_params = save_dict['agent_params'][self.num_predators:]

        for i, params in zip(range(self.num_preys), prey_params):
            self.preys[i].load_params_without_optims(params)

        for p in self.preys:
            p.policy.eval()
            p.target_policy.eval()

"""
get other agents' gaussian policy,
calculate Q_total and Ai
diffusion score from singel agent miu_i(a|s)

Final loss is E_pi^i_new, pi^ -i [A^i + log miu_i] d/ pi_i
"""
class SEQ_SRPO(object):
    def __init__(self,
                 )
        

    def run():


class VD_SRPO(object):
    def __init__(self,
                 )
        

    def run():
        


# class JAL_SRPO(object):
#     def __init__(
#         self, 
#         agent_init_params, # state & action dim
#         agent_max_actions, 
#         alg_types, 
#         denoise_steps=20,
#         device = 'cpu',
#         adv_init_params=None,
#         gamma=0.99, # RL discount gamma
#         tau=0.01,  # 这个 tau 是用来 target network soft update的
#         lr=0.01, 
#         hidden_dim=64, 
#         discrete_action=False, 
#         env_id=None,
#         batch_size = 100,
#         **kwargs
#     ):
#         self.env_id = env_id
#         self.is_mamujoco = True if self.env_id == 'HalfCheetah-v2' else False

#         assert (ma == agent_max_actions[0] for ma in agent_max_actions)
#         self.max_action = agent_max_actions[0]
#         self.min_action = -self.max_action
#         self.hidden_dim = hidden_dim  # 64 for DDPG

#         self.nagents = len(alg_types)
#         self.alg_types = alg_types
#         self.tau = tau
  
#         self.agent_init_params = agent_init_params
#         # in JAL, the state and action dim are multiplex by agent numbers
#         self.state_dim = self.agent_init_params[0]['state_dim'] * self.nagents
#         self.action_dim = self.agent_init_params[0]['action_dim'] * self.nagents
#         self.device = device
        
#         self.gamma = gamma
#         self.lr = lr
#         self.discrete_action = discrete_action

#         self.T = denoise_steps  # if JAL, T == 20+
#         self.batch_size = batch_size

#         self.agents = [SRPO(
#             state_dim = self.state_dim,
#             action_dim = self.action_dim,
#             max_action = self.max_action,
#             device = self.device,
#             discount = self.gamma,
#             tau=self.tau,
#             max_q_backup=False,
#             beta_schedule='linear',
#             n_timesteps=self.T,
#             ema_decay=0.995,
#             step_start_ema=1000,
#             update_ema_every=5,
#             lr=self.lr,
#             lr_decay=False,
#             lr_maxt=1000,
#             grad_norm=1.0
#             )]


#         if self.env_id in ['simple_tag', 'simple_world']:
#             self.num_predators = len(agent_init_params)
#             self.num_preys = len(adv_init_params)

#             self.preys = [DDPGAgent(lr=lr, discrete_action=self.discrete_action, hidden_dim=self.hidden_dim, **params) for params in adv_init_params]

#         self.niter = 0

        
#         for k, v in kwargs.items():
#             setattr(self, k, v)
        

#     @property
#     def policies(self):
#         return [a.actor for a in self.agents]
#     # self.policy = Diffusion()
#     # action = self.policy.forward(obs)

#     @property
#     def target_policies(self):
#         return [a.ema_model for a in self.agents]
#     # self.ema_model = copy.deepcopy(self.actor)
#     # 但是这个并不用作 target policy，没有 soft update，soft只作用于 diffusion critic


#     def step(self, observations, explore=False):
#         """
#         Take a step forward in environment with all agents
#         Inputs:
#             observations: List of observations for each agent
#             explore (boolean): Whether or not to add exploration noise
#         Outputs:
#             actions: List of actions for each agent
#         """

#         # input torch_obs is [torch(1,18), torch(1,18), torch(1,18)]
#         # output actions is [array(2), array(2), array(2)]
#         # for JAL, one diff model generate all agents actions

#         if self.env_id in ['simple_world', 'simple_tag']:
#             obs_predator = observations[:self.num_predators]
#             obs_predator = torch.cat(obs_predator, dim=1)
#             obs_prey = observations[self.num_predators:]

#             predator_actions = self.agents[0].sample_action(obs_predator) # input tensor[1,obs_dim * n] output array[1, act_dim * n]
#             actions = np.split(predator_actions, self.num_predators, axis=0)

#             for i, obs in zip(obs_prey):
#                 prey_action = self.preys[i].step(obs, explore=False)
#                 actions.append(prey_action)
#         else:
#             observations = torch.cat(observations, dim=1)
#             joint_actions = self.agents[0].sample_action(observations)
#             actions = np.split(joint_actions, self.nagents, axis=0)

#         assert len(actions[0]) == self.agent_init_params[0]['action_dim']

#         return actions
    

    
#     def update(self, sample, t):

#         JAL_agent = self.agents[0]

#         # input data of JAL is jointly, which needs to be concated as [batch_size * (dim*agent_numbers)]
#         jal_obs = [sample_i['obs'] for sample_i in sample]
#         jal_obs = torch.cat(jal_obs, dim=1)
#         assert jal_obs.size()[0] == self.batch_size

#         jal_act = [sample_i['action'] for sample_i in sample]  # nagents * batch_size * act_dim
#         jal_act = torch.cat(jal_act, dim=1) # batch_size * (nagents*act_dim)

#         jal_rew = sample[0]['rewards']  # [batch_size]
#         jal_done = sample[0]['done'] # [batch_size]

#         jal_next_obs = [sample_i['next_obs'] for sample_i in sample]
#         jal_next_obs = torch.cat(jal_next_obs, dim=1)

#         jal_next_act = [sample_i['next_action'] for sample_i in sample]  
#         jal_next_act = torch.cat(jal_next_act, dim=1) 

#         jal_sample_dict = {"obs": jal_obs,
#                            "action": jal_act,
#                            "rewards": jal_rew,
#                            "next_obs": jal_next_obs,
#                            "done": jal_done,
#                            "next_action": jal_next_act,}
        
#         if self.is_mamujoco:
#             jal_state = [sample_i['state'] for sample_i in sample]
#             jal_state = torch.cat(jal_state, dim=1)
            
#             jal_next_state = [sample_i['next_state'] for sample_i in sample]
#             jal_next_state = torch.cat(jal_next_state, dim=1)

#             jal_sample_dict.update({"state": jal_state})
#             jal_sample_dict.update({"next_state": jal_next_state})
        
#         metric = JAL_agent.train(jal_sample_dict, batch_size=self.batch_size, log_writer=None, is_mujoco = self.is_mamujoco)
        
#         """ logging metric """
#         if t % self.logging_interval == 0 and not self.no_log:
#             dic = {}
#             dic.update({"BC loss agent"+str("JAL"):metric['bc_loss']})
#             dic.update({"QK loss agent"+str("JAL"):metric['ql_loss']})
#             dic.update({"Actor loss agent"+str("JAL"):metric['actor_loss']})
#             dic.update({"Critic loss agent"+str("JAL"):metric['critic_loss']})
            
#             log_and_print(list(dic.keys()), list(dic.values()), t, multi=True)

#     # prepare train() or eval() 
#     def prep_training(self, device='cpu'):

#         for a in self.agents:
#             a.model.train()
#             a.actor.train()
#             a.critic.train()

#         fn = lambda x: x.to(device)   
#         for a in self.agents:
#             a.model = fn(a.model)
#             a.actor = fn(a.actor)
#             a.critic = fn(a.critic)

#         if self.env_id in ['simple_tag', 'simple_world']:
#             for p in self.preys:
#                 p.policy = fn(p.policy)
#                 p.target_policy = fn(p.target_policy)

#     def prep_rollouts(self, device='cpu'):
#         for a in self.agents:
#             a.model.eval()
#             a.actor.eval()
#             a.critic.eval()

#         fn = lambda x: x.to(device)

#         for a in self.agents:
#             a.model = fn(a.model)
#             a.actor = fn(a.actor)

#         if self.env_id in ['simple_tag', 'simple_world']:
#             for p in self.preys:
#                 p.policy = fn(p.policy)


#     @classmethod
#     def init_from_env(cls, env, env_id, env_info=None, agent_alg="diffusion", adversary_alg="ddpg",
#                        gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
#                        batch_size=None, denoise_steps = 20, **kwargs):
#         """
#         Instantiate instance of this class from multi-agent environment
#         """

#         # create n trainable agents without prey, alg_types = ['diff', 'diff', 'diff']
#         if env_id in ['simple_tag', 'simple_world']:
#             alg_types = [agent_alg for atype in env.agent_types if atype == 'adversary']
#         elif env_id in ['simple_spread']:
#             alg_types = [agent_alg for atype in env.agent_types]
#         elif env_id in ['HalfCheetah-v2']:
#             alg_types = [agent_alg for _ in range(env_info['n_agents'])]

#         agent_init_params = []
#         all_n_actions = []
#         agent_max_actions = []
#         adv_init_params = []

#         # make agent_init_params, adv_init_params, agent_max_actions, all_n_actions
#         if env_id == 'HalfCheetah-v2':
#             for agent_idx in range(len(alg_types)):
#                 acsp = env_info['action_spaces'][agent_idx]
#                 num_in_pol = env_info['obs_shape']
#                 num_out_pol = acsp.shape[0]

#                 agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                
#                 agent_max_actions.append(acsp.high[0])
#                 all_n_actions.append(acsp.shape[0])
#         else:
#             for acsp, obsp, agent_type in zip(env.action_space, env.observation_space, env.agent_types):
#                 num_in_pol = obsp.shape[0]
#                 num_out_pol = acsp.shape[0]
#                 num_in_critic = num_in_pol + num_out_pol

#                 if env_id in ['simple_spread']:
#                     agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
#                     agent_max_actions.append(acsp.high[0])
#                 else:
#                     if agent_type == 'adversary':  # adversary 是猎人，agent 是猎物
#                         agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
#                         agent_max_actions.append(acsp.high[0])
#                     elif agent_type == 'agent':
#                         adv_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})

#                 all_n_actions.append(acsp.shape[0])

#             for i in range(1, len(all_n_actions)):
#                 assert (all_n_actions[i] == all_n_actions[0])  # 同时包括了 predator 和 prey 的动作维度


#         init_dict = {
#             'agent_init_params': agent_init_params,
#             'agent_max_actions': agent_max_actions,
#             'alg_types': alg_types,
#             'denoise_steps': denoise_steps,
#             'device': 'cpu',
#             'adv_init_params': adv_init_params,
#             'gamma': gamma, 
#             'tau': tau,
#             'lr': lr,
#             'hidden_dim': hidden_dim,
#             'discrete_action': False,
#             'env_id': env_id,
#             'batch_size': batch_size,   
#         }

#         # if have additional params in kwargs
#         init_dict.update(kwargs)
#         instance = cls(**init_dict)
#         instance.init_dict = init_dict
        
#         return instance


#     def load_pretrained_preys(self, filename):
#         if not torch.cuda.is_available():
#             save_dict = torch.load(filename, map_location=torch.device('cpu'))
#         else:
#             save_dict = torch.load(filename)

#         if self.env_id in ['simple_tag', 'simple_world']:
#             prey_params = save_dict['agent_params'][self.num_predators:]

#         for i, params in zip(range(self.num_preys), prey_params):
#             self.preys[i].load_params_without_optims(params)

#         for p in self.preys:
#             p.policy.eval()
#             p.target_policy.eval()



# class QMIX_SRPO(object):
#     def __init__():
            
# class SEQ_SRPO(object):
#     def __init__():
            






def log_and_print(key, value, t, multi=False):
    if multi:
        print("t:", t, end=" | ")
        for i in range(len(key)):
            end = " | " if i < len(key) - 1 else "\n"
            print("{}: {:.3f}".format(key[i], value[i][0]), end=end)
            log_value(key[i], value[i][0], t)    
    else:
        print("t:{}, {}: {:.3f}".format(t, key, value))
        log_value(key, value, t)


