# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
from .model import *

# SRPO for IND & JAL_MASRPO
class SRPO(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=args.t_GassProj_dims, args=args)
        # self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=args.learning_rates)
        self.diffusion_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.diffusion_optimizer, T_max=2000000, eta_min=1e-5)
        self.SRPO_policy = Dirac_Policy(output_dim, input_dim-output_dim, layer=args.policy_layer).to(args.device)
        self.SRPO_policy_optimizer = torch.optim.Adam(self.SRPO_policy.parameters(), lr=args.dilac_lr)
        self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.SRPO_policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.q.append(IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
    
    # 这里要适配一下 MARL dataset
    def update_SRPO_policy(self, data):
        s = data['s']        
        self.diffusion_behavior.eval()
        a = self.SRPO_policy(s)   # dilac policy
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        # random noising time t
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]   # alpha 从 512 unsqueeze成 512,1；方便按位相乘
        # add noise to policy action, generate a_t

        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()  # diffusion model prediction
            if "noise" in self.args.WT:
                episilon = episilon - z

        if "VDS" in self.args.WT:
            wt = std ** 2
        elif "stable" in self.args.WT:
            wt = 1.0
        elif "score" in self.args.WT:
            wt = alpha_t / std
        else:
            assert False

        detach_a = a.detach().requires_grad_(True)  # dilac policy detach
        qs = self.q[0].q0_target.both(detach_a , s)  # Q(s, a) 
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)


        # TODO be aware that there is a small std gap term here, this seem won't affect final performance though
        # guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach() * std[..., None]
        guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        # dq/da gradient

        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        # Q gradient + diffusion gradient (noise*sigma)
        loss = (episilon * a).sum(-1) * wt - (guidance * a).sum(-1) * self.args.beta
        # max Q - epsilon = min epsilon - Q
        loss = loss.mean()
        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()
        self.diffusion_behavior.train()

        error_a = torch.mean(data['a'] - a)

        if self.args.env_id == 'bandit':
            return loss, episilon, guidance, error_a, a
        else:
            return loss, episilon, guidance, error_a
    

# SRPO for Seq_MASRPO, which loads joint Q and ind diffusion

class SRPO_CTDE(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        # diffusion model is individual
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=args.t_GassProj_dims, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=args.learning_rates)
        self.diffusion_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.diffusion_optimizer, T_max=2000000, eta_min=1e-5)
        # self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        # SRPO dilac policy is individual
        self.SRPO_policy = Dirac_Policy(output_dim, input_dim-output_dim, layer=args.policy_layer).to(args.device)
        self.SRPO_policy_optimizer = torch.optim.Adam(self.SRPO_policy.parameters(), lr=3e-4)
        self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.SRPO_policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        
        # input = state + action, output = action
        # here we load a centralized/advantage Q value with IQL (can be replaced by ICQ/OMAR)
        n_agent_numbers = len(args.alg_types)
        self.q = []        
        self.q.append(IQL_Critic(adim=output_dim*n_agent_numbers, sdim=(input_dim-output_dim)*n_agent_numbers, args=args))
        # for mamujoco halfcheetah, state is 6 and action is 3*2
        # input joint s, output joint a

    def update_SRPO_policy(self, data, agent_id):
        s = data['s']        
        s_joint = data['s_joint'] # 用于计算Q值的condition
        a_joint = data['a_joint'] # 用于计算Q值的action

        self.diffusion_behavior.eval()
        a = self.SRPO_policy(s)
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        # random noising time t
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        # add noise to policy action, generate a_t

        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s).detach()  # diffusion model prediction
            if "noise" in self.args.WT:
                episilon = episilon - z

        if "VDS" in self.args.WT:
            wt = std ** 2
        elif "stable" in self.args.WT:
            wt = 1.0
        elif "score" in self.args.WT:
            wt = alpha_t / std
        else:
            assert False

        detach_a = a.detach().requires_grad_(True)
        a_joint[agent_id] = detach_a

        detach_a_joint = torch.cat(a_joint, dim=1)

        # Dilac policy action and Q(s, a) 这里用的是JAL Q(state_tot, action_tot) s要改成concate的
        qs = self.q[0].q0_target.both(detach_a_joint, s_joint)  
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)


        guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        # dq/da gradient

        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        # (Q tot 对 a_i 求梯度 + score i)

        loss = (episilon * a).sum(-1) * wt - (guidance * a).sum(-1) * self.args.beta

        # max Q - epsilon = min epsilon - Q
        loss = loss.mean()
        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()
        self.diffusion_behavior.train()

        error_a = torch.mean(data['a'] - a)

        if self.args.env_id == 'bandit':
            return loss, episilon, guidance, error_a, a
        else:
            return loss, episilon, guidance, error_a
        
# 第二个agent的policy维度需要调整，policy维度并没有变化
class SRPO_ssd(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        # diffusion model is individual
        # input state+2action, output action
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=args.t_GassProj_dims, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=args.learning_rates)
        self.diffusion_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.diffusion_optimizer, T_max=2000000, eta_min=1e-5)
        # self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        # SRPO dilac policy is individual
        self.SRPO_policy = Dirac_Policy(output_dim, input_dim-2*output_dim, layer=args.policy_layer).to(args.device)
        # input=s+2a, output=a
        self.SRPO_policy_optimizer = torch.optim.Adam(self.SRPO_policy.parameters(), lr=3e-4)
        self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.SRPO_policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        
        # input = state + action, output = action
        # here we load a centralized/advantage Q value with IQL (can be replaced by ICQ/OMAR)
        n_agent_numbers = len(args.alg_types)
        self.q = []        
        self.q.append(IQL_Critic(adim=output_dim*n_agent_numbers, sdim=(input_dim-2*output_dim)*n_agent_numbers, args=args))
        # for mamujoco halfcheetah, state is 6 and action is 3*2
        # input joint s, output joint a

    def update_SRPO_policy(self, data, agent_id):
        s = data['s']        
        s_joint = data['s_joint'] # 用于计算Q值的condition
        a_joint = data['a_joint'] # 用于计算Q值的action

        self.diffusion_behavior.eval()
    
        a = self.SRPO_policy(s)
        
            
        t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
        # random noising time t
        alpha_t, std = self.marginal_prob_std(t)
        z = torch.randn_like(a)
        perturbed_a = a * alpha_t[..., None] + z * std[..., None]
        # add noise to policy action, generate a_t


        s_condition = torch.cat((s, a_joint[0]), dim=1).to(self.args.device)
        # 第二个 agent score 维度是s+a

        with torch.no_grad():
            episilon = self.diffusion_behavior(perturbed_a, t, s_condition).detach()  # diffusion model prediction
            if "noise" in self.args.WT:
                episilon = episilon - z

        if "VDS" in self.args.WT:
            wt = std ** 2
        elif "stable" in self.args.WT:
            wt = 1.0
        elif "score" in self.args.WT:
            wt = alpha_t / std
        else:
            assert False

        detach_a = a.detach().requires_grad_(True)
        a_joint[agent_id] = detach_a

        detach_a_joint = torch.cat(a_joint, dim=1)

        # Dilac policy action and Q(s, a) 这里用的是JAL Q(state_tot, action_tot) s要改成concate的
        qs = self.q[0].q0_target.both(detach_a_joint, s_joint)  
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)


        guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
        # dq/da gradient

        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        # (Q tot 对 a_i 求梯度 + score i)

        loss = (episilon * a).sum(-1) * wt - (guidance * a).sum(-1) * self.args.beta

        # max Q - epsilon = min epsilon - Q
        loss = loss.mean()
        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()
        self.diffusion_behavior.train()

        error_a = torch.mean(data['a'] - a)

        if self.args.env_id == 'bandit':
            return loss, episilon, guidance, error_a, a
        else:
            return loss, episilon, guidance, error_a
    

class MASRPO_Behavior(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=args.t_GassProj_dims, args=args)
        self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=args.learning_rates)
        if args.lr_anneal:
            self.diffusion_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.diffusion_optimizer, T_max=2000000, eta_min=1e-5)

        # self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=64, args=args)
        # self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
        # self.diffusion_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.diffusion_optimizer, T_max=1500000, eta_min=0.)

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.device = args.device
    
    def update_behavior(self, data):
        self.step += 1
        all_a = data['action'].to(self.device)
        all_s = data['obs'].to(self.device)
        # Mujuco use obs instead of state

        # Update diffusion behavior
        self.diffusion_behavior.train()

        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1. - 1e-3) + 1e-3  
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        episilon = self.diffusion_behavior(perturbed_x, random_t, all_s)
        loss = torch.mean(torch.sum((episilon - z)**2, dim=(1,)))
        self.loss = loss

        self.diffusion_optimizer.zero_grad()
        loss.backward()  
        self.diffusion_optimizer.step()
        self.diffusion_lr_scheduler.step()

        return loss
        

# used in pretrain_critic.py
 
class MASRPO_IQL(nn.Module):
    def __init__(self, input_dim, output_dim, args=None):
        super().__init__()
        self.deter_policy = Dirac_Policy(output_dim, input_dim-output_dim).to(args.device)
        self.deter_policy_optimizer = torch.optim.Adam(self.deter_policy.parameters(), lr=3e-4)
        self.deter_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.deter_policy_optimizer, T_max=2000000, eta_min=0.)

        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.q.append(IQL_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
    
    def update_iql(self, data):
        a = data['action']
        # if mujoco, use obs as state 
        s = data['obs']
        self.q[0].update_q0(data)
        
        # evaluate iql policy part, can be deleted
        with torch.no_grad():
            target_q = self.q[0].q0_target(a, s).detach()
            v = self.q[0].vf(s).detach()
        adv = target_q - v
        temp = 10.0 if "maze" in self.args.env_id else 3.0
        exp_adv = torch.exp(temp * adv.detach()).clamp(max=100.0)        

        policy_out = self.deter_policy(s)
        bc_losses = torch.sum((policy_out - a)**2, dim=1)   # pi(a|s) - a_data
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)   # (a - a0) * e^( 10* (Q-V) )
        self.deter_policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.deter_policy_optimizer.step()
        self.deter_policy_lr_scheduler.step()
        self.policy_loss = policy_loss

        mean_bc_losses = torch.mean(bc_losses)

        return policy_loss, mean_bc_losses



def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL_Critic(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        self.q0 = TwinQ(adim, sdim, layers=args.q_layer).to(args.device)
        self.q0_target = copy.deepcopy(self.q0).to(args.device)

        self.vf = ValueFunction(sdim).to(args.device)
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=args.iql_critic_lr)   # 3e-4 有些任务爆炸，试试更小的lr
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=args.iql_critic_lr)

        self.discount = 0.99
        self.args = args
        if "maze" in args.env_id:
            self.tau = 0.9
        elif "simple" in args.env_id:
            self.tau = 0.8
        else:
            self.tau = 0.7
        self.clip_degree = 0.5  # grad clipping

    def update_q0(self,data):
        s = data["obs"]
        a = data["action"]
        r = data["rewards"]
        s_ = data["next_obs"]
        d = data["done"]
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v

        # adv = adv.clamp(max=100.0)   # 新增clamp稳定训练

        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()

        # # 为了避免梯度爆炸，新增clip，原始IQL和SRPO中没有
        # torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.clip_degree)  # 添加梯度裁剪

        self.v_optimizer.step()
        
        # Update Q function
        # 这里做了修改，1-done是256维度，乘以后面的256，1维度会错误的变成256，256
        targets = r.unsqueeze(1) + (1. - d.float()).unsqueeze(1) * self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        self.v = v.mean()
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()

        # # 为了避免梯度爆炸，新增clip，原始IQL和SRPO中没有
        # torch.nn.utils.clip_grad_norm_(self.q0.parameters(), self.clip_degree)  # 添加梯度裁剪

        self.q_optimizer.step()
        self.v_loss = v_loss
        self.q_loss = q_loss
        self.q = target_q.mean()
        self.v = next_v.mean()
        # Update target
        update_target(self.q0, self.q0_target, 0.005)        





# # SRPO for SEQ_SRPO_others, 考虑了其他人的动作期望在loss里

# class SRPO_CTDE_others(nn.Module):
#     def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
#         super().__init__()
#         # diffusion model is individual
#         self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=args.t_GassProj_dims, args=args)
#         self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=args.learning_rates)
#         self.diffusion_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.diffusion_optimizer, T_max=2000000, eta_min=1e-5)
#         # self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
#         # SRPO dilac policy is individual
#         self.SRPO_policy = Dirac_Policy(output_dim, input_dim-output_dim, layer=args.policy_layer).to(args.device)
#         self.SRPO_policy_optimizer = torch.optim.Adam(self.SRPO_policy.parameters(), lr=3e-4)
#         self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.SRPO_policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

#         self.marginal_prob_std = marginal_prob_std
#         self.args = args
#         self.output_dim = output_dim
#         self.step = 0
        
#         # input = state + action, output = action
#         # here we load a centralized/advantage Q value with IQL (can be replaced by ICQ/OMAR)
#         n_agent_numbers = len(args.alg_types)
#         self.q = []        
#         self.q.append(IQL_Critic(adim=output_dim*n_agent_numbers, sdim=(input_dim-output_dim)*n_agent_numbers, args=args))
#         # for mamujoco halfcheetah, state is 6 and action is 3*2
#         # input joint s, output joint a

#     def update_SRPO_policy(self, data, agent_id):
#         s = data['s']        
#         s_joint = data['s_joint'] # 用于计算Q值的condition
#         a_joint = data['a_joint'] # 用于计算Q值的action

#         self.diffusion_behavior.eval()
#         a = self.SRPO_policy(s)
#         t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
#         # random noising time t
#         alpha_t, std = self.marginal_prob_std(t)
#         z = torch.randn_like(a)
#         perturbed_a = a * alpha_t[..., None] + z * std[..., None]
#         # add noise to policy action, generate a_t

#         with torch.no_grad():
#             episilon = self.diffusion_behavior(perturbed_a, t, s).detach()  # diffusion model prediction
#             if "noise" in self.args.WT:
#                 episilon = episilon - z

#         if "VDS" in self.args.WT:
#             wt = std ** 2
#         elif "stable" in self.args.WT:
#             wt = 1.0
#         elif "score" in self.args.WT:
#             wt = alpha_t / std
#         else:
#             assert False

#         detach_a = a.detach().requires_grad_(True)
#         a_joint[agent_id] = detach_a

#         # 提取其他人当前策略下的联合动作用于计算期望
#         other_actions = a_joint[:agent_id] + a_joint[agent_id+1:]
#         joint_other_actions = other_actions[0]
#         for other in other_actions[1:]:
#             joint_other_actions = torch.mul(joint_other_actions, other)
#         assert joint_other_actions.shape == a.shape
#         # other_actions_tuple = tuple(other_actions)
#         # joint_other_actions = torch.mul(*other_actions_tuple)

#         detach_a_joint = torch.cat(a_joint, dim=1)

#         # Dilac policy action and Q(s, a) 这里用的是JAL Q(state_tot, action_tot) s要改成concate的
#         qs = self.q[0].q0_target.both(detach_a_joint, s_joint)  
#         q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
#         self.SRPO_policy.q = torch.mean(q)


#         guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
#         # dq/da gradient

#         if self.args.regq:
#             guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
#             guidance = guidance / guidance_norm

#         # (Q tot 对 a_i 求梯度 + score i), 这里额外乘以其他人的dilac policy as expectation
#         loss = (episilon * a * joint_other_actions).sum(-1) * wt - (guidance * a * joint_other_actions).sum(-1) * self.args.beta

#         # max Q - epsilon = min epsilon - Q
#         loss = loss.mean()
#         self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         self.SRPO_policy_optimizer.step()
#         self.SRPO_policy_lr_scheduler.step()
#         self.diffusion_behavior.train()

#         error_a = torch.mean(data['a'] - a)

#         return loss, error_a





# # SRPO for Seq_MASRPO, which loads joint Q and ind diffusion

# class SRPO_SEQ(nn.Module):
#     def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
#         super().__init__()
#         # diffusion model is individual
#         self.diffusion_behavior = ScoreNet_IDQL(input_dim, output_dim, marginal_prob_std, embed_dim=args.t_GassProj_dims, args=args)
#         self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=args.learning_rates)
#         self.diffusion_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.diffusion_optimizer, T_max=2000000, eta_min=1e-5)
#         # self.diffusion_optimizer = torch.optim.AdamW(self.diffusion_behavior.parameters(), lr=3e-4)
#         # SRPO dilac policy is individual
#         self.SRPO_policy = Dirac_Policy(output_dim, input_dim-output_dim, layer=args.policy_layer).to(args.device)
#         self.SRPO_policy_optimizer = torch.optim.Adam(self.SRPO_policy.parameters(), lr=3e-4)
#         self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.SRPO_policy_optimizer, T_max=args.n_policy_epochs * 10000, eta_min=0.)

#         self.marginal_prob_std = marginal_prob_std
#         self.args = args
#         self.output_dim = output_dim
#         self.step = 0
        
#         # input = state + action, output = action
#         # here we load a centralized/advantage Q value with IQL (can be replaced by ICQ/OMAR)
#         n_agent_numbers = len(args.alg_types)
#         self.q = []        
#         self.q.append(IQL_Critic(adim=output_dim*n_agent_numbers, sdim=(input_dim-output_dim)*n_agent_numbers, args=args))
#         # for mamujoco halfcheetah, state is 6 and action is 3*2
#         # input joint s, output joint a

#     # 这里要适配一下 MARL dataset
#     def update_SRPO_policy(self, data, prefix_policy, suffix_policy):
#         s = data['s']        
#         s_joint = data['s_joint'] # 用于计算Q值的condition

#         self.diffusion_behavior.eval()
#         [diff.diffusion_behavior.eval() for diff in prefix_policy]
#         [diff.diffusion_behavior.eval() for diff in suffix_policy]

#         a = self.SRPO_policy(s)

#         a_pre = []
#         a_suf = []
#         if len(prefix_policy) <= 0:
#             pass
#         else:
#             for pre in prefix_policy:
#                 a_pre.append(pre.SRPO_policy(s))
#         if len(suffix_policy) <=0:
#             pass
#         else:
#             for suf in suffix_policy:
#                 a_suf.append(suf.SRPO_policy(s))


#         t = torch.rand(a.shape[0], device=s.device) * 0.96 + 0.02
#         # random noising time t
#         alpha_t, std = self.marginal_prob_std(t)
#         z = torch.randn_like(a)
#         perturbed_a = a * alpha_t[..., None] + z * std[..., None]
#         # add noise to policy action, generate a_t

#         perturbed_a_pre = []
#         perturbed_a_suf = []

#         if len(prefix_policy) <= 0:
#             pass
#         else:
#             for pre in range(len(prefix_policy)):
#                 p_a_pre = a_pre[pre] * alpha_t[..., None] + z * std[..., None]
#                 perturbed_a_pre.append(p_a_pre)
#         if len(suffix_policy) <=0:
#             pass
#         else:
#             for suf in range(len(suffix_policy)):
#                 p_a_suf = a_suf[suf] * alpha_t[..., None] + z * std[..., None]
#                 perturbed_a_suf.append(p_a_suf)

#         with torch.no_grad():
#             episilon = self.diffusion_behavior(perturbed_a, t, s).detach()  # diffusion model prediction
#             epi_pre = []
#             epi_suf = []
#             if len(prefix_policy) <= 0:
#                 pass
#             else:
#                 for pre in range(len(prefix_policy)):
#                     epi_pre_i = prefix_policy[pre].diffusion_behavior(p_a_pre, t, s).detach()
#                     epi_pre.append(epi_pre_i)
#             if len(suffix_policy) <=0:
#                 pass
#             else:
#                 for suf in range(len(suffix_policy)):
#                     epi_suf_i = suffix_policy[suf].diffusion_behavior(p_a_suf, t, s).detach()
#                     epi_suf.append(epi_suf_i)
            

#             if "noise" in self.args.WT:
#                 episilon = episilon - z
#                 epi_pre = epi_pre - z
#                 epi_suf = epi_suf - z

#         if "VDS" in self.args.WT:
#             wt = std ** 2
#         elif "stable" in self.args.WT:
#             wt = 1.0
#         elif "score" in self.args.WT:
#             wt = alpha_t / std
#         else:
#             assert False

#         # here we need to consider prefix agents' new policy and actions a'
#         detach_a = a.detach().requires_grad_(True)
#         """ others' a = xxx, a_joint = detach_a + others' a"""
#         # a_pre = torch.tensor(a_pre)
#         # a_suf = torch.tensor(a_suf)

#         a_joint = a_pre+[detach_a]+a_suf
#         detach_a_joint = torch.cat(a_joint, dim=1)

#         # Dilac policy action and Q(s, a) 这里用的是JAL Q(state_tot, action_tot) s要改成concate的
#         qs = self.q[0].q0_target.both(detach_a_joint, s_joint)  
#         q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
#         self.SRPO_policy.q = torch.mean(q)


#         guidance =  torch.autograd.grad(torch.sum(q), detach_a)[0].detach()
#         # dq/da gradient

#         if self.args.regq:
#             guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
#             guidance = guidance / guidance_norm

#         # 这里可以把其他diff的score给进来，但是因为detach了所以是个常数，不影响梯度
#         # guidance只对当前的a保留梯度，
#         epi_all = epi_pre + [episilon] + epi_suf
#         episilon_all = torch.cat(epi_all, dim=1)
#         loss = (episilon_all * detach_a_joint).sum(-1) * wt - (guidance * a).sum(-1) * self.args.beta

#         # max Q - epsilon = min epsilon - Q
#         loss = loss.mean()
#         self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         self.SRPO_policy_optimizer.step()
#         self.SRPO_policy_lr_scheduler.step()
#         self.diffusion_behavior.train()
#         [diff.diffusion_behavior.train() for diff in prefix_policy]
#         [diff.diffusion_behavior.train() for diff in suffix_policy]

#         rtn_epi_all = torch.sum(episilon_all, dim=0)
#         rtn_guide = torch.sum(guidance, dim=0)

#         return loss, rtn_epi_all, rtn_guide
    
# used in pretrain_behavior.py, no IQL_critic module
