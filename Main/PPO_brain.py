import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):
    def __init__(self, n_states,n_actions, bound):
        super(ActorNet, self).__init__()
        self.n_states = n_states
        self.n_actions=n_actions
        self.bound = bound
        Neurons=self.n_states*8*3
        
        self.layer = nn.Sequential(
            nn.Linear(self.n_states, Neurons),
            nn.ReLU(),
            nn.Linear(Neurons, Neurons//2),
            nn.ReLU(),
            nn.Linear(Neurons//2, Neurons//4),
            nn.ReLU(),
            nn.Linear(Neurons//4, Neurons//8),
            nn.ReLU(),
        )
        self.mu_out = nn.Linear(Neurons//8, n_actions)
        self.sigma_out = nn.Linear(Neurons//8, n_actions)

    def forward(self, x):
        x = self.layer(x)
        mu = torch.tanh(self.mu_out(x))*self.bound#均值
        sigma = F.softplus(self.sigma_out(x))#標準差
        return mu, sigma

class CriticNet(nn.Module):
    def __init__(self, n_states):
        super(CriticNet, self).__init__()
        self.n_states = n_states
        Neurons=self.n_states*8*3

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, Neurons//2),
            nn.ReLU(),
            nn.Linear(Neurons//2, Neurons//4),
            nn.ReLU(),
            nn.Linear(Neurons//4, Neurons//8),
            nn.ReLU(),
            nn.Linear(Neurons//8, 1)
        )

    def forward(self, x):
        v = self.layer(x)
        return v
    
class PPO:
    def __init__(self, n_states, n_actions, bound, PPO_parameter, device):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.PPO_parameter=PPO_parameter
        self.lr = 1e-5  # 預設學習率，會被 set_learning_rate() 覆寫
        self.gamma = PPO_parameter.gamma
        self.epsilon = PPO_parameter.epsilon
        self.a_update_steps = PPO_parameter.a_update_steps
        self.c_update_steps = PPO_parameter.c_update_steps
        self.device=device
        self.lmbda = 0.9  # GAE優勢函數的縮放因子
        
        self._build()

    def _build(self):
        self.actor_model = ActorNet(self.n_states, self.n_actions,self.bound)
        self.actor_old_model = ActorNet(self.n_states, self.n_actions,self.bound)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=self.lr)

        self.critic_model = CriticNet(self.n_states)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=self.lr)
        
    def choose_action(self, states):
        states = torch.FloatTensor(states)
        mu, sigma = self.actor_model(states)
        dist = torch.distributions.Normal(mu, sigma)
        action=np.clip(dist.sample(),-self.bound, self.bound).reshape(-1)# 轉換為Python列表
        return action
    
    def actor_learn(self, states, actions, advantage):
        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)
        prob=pi.log_prob(actions)
        
        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)
        old_prob=old_pi.log_prob(actions)
        
        if torch.any(prob < -6) or torch.any(old_prob < -6):
            return
         
        ratio=torch.exp(prob-old_prob)
        advantage_sign = torch.sign(advantage)
            
        surr1=ratio*advantage_sign
        surr2=torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)*advantage_sign
        min_surr=torch.min(surr1,surr2)*advantage_sign
        
        ratio_product=min_surr.prod(dim=1, keepdim=True)
        loss = -torch.mean( ratio_product*advantage)
        #print("loss : ", loss,"\n")
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        v = self.critic_model(states)

        loss_func = nn.MSELoss()
        loss = loss_func(v, targets)
        
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def cal_target(self, rewards, next_states):
        next_states_target = self.critic_model(next_states).detach()                   # torch.Size([batch, 1])
        target_list = rewards + self.gamma * next_states_target                         # torch.Size([batch, 1])
        return target_list

    def cal_advantage(self, states, targets):
        v = self.critic_model(states)                             # torch.Size([batch, 1])
        advantage = (targets - v).detach()                  # torch.Size([batch, 1])
        return advantage
    
    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())        # 首先更新舊模型
        advantage = self.cal_advantage(states, targets)

        for i in range(self.a_update_steps):                      # 更新多次
            self.actor_learn(states, actions, advantage)

        for i in range(self.c_update_steps):                      # 更新多次
            self.critic_learn(states, targets)
        
    def training(self, replay_buffer):
        #搬到GPU
        self.actor_model.to(self.device)
        self.actor_old_model.to(self.device)
        self.critic_model.to(self.device)
        for t in range (self.PPO_parameter.n_round_batch):
            batch = replay_buffer.sample(self.PPO_parameter.mini_batch)
            if len(batch)==0:break
            states, actions, rewards, next_states = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).to(self.device).reshape(-1,1)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            targets=self.cal_target(rewards, next_states)
            self.update(states, actions, targets)
        #從GPU搬回來
        self.actor_model.to('cpu')
        self.actor_old_model.to('cpu')
        self.critic_model.to('cpu')

    def set_learning_rate(self, lr):
        """動態更新學習率。"""
        self.lr = lr
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # 擴展緩沖區
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity  # 循環存儲

    def sample(self, batch_size):
        if len(self.buffer)<batch_size:
            batch_size=len(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)