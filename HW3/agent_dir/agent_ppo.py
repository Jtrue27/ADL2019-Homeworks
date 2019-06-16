import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Categorical
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class PPO(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_size):
        super(PPO, self).__init__()
        self.affine = nn.Linear(obs_shape, hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(obs_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_shape),
            nn.Softmax(dim=-1),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),

        )
        
        
    def forward(self, state, action=None, evaluate=False):
        
        raise NotImplementedError


class AgentPPO(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PPO(obs_shape = self.env.observation_space.shape[0],
                               act_shape= self.env.action_space.n,
                               hidden_size=64).to(device)
        self.model_old = PPO(obs_shape = self.env.observation_space.shape[0],
                               act_shape= self.env.action_space.n,
                               hidden_size=64).to(device)
        if args.test_ppo:
            self.load('ppo.cpt')
        # discounted reward
        self.gamma = 0.99 
        # i am not sure beta
        self.betas = (0.9, 0.999) 
        self.eps_clip = 0.2
        self.K_epochs = 5
        self.steps = 0 
        self.MseLoss = nn.MSELoss()



        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3,betas=self.betas)
        # saved rewards and actions
        # self.rewards, self.saved_actions = [], []
        # self.saved_log_probs=[]
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
    
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model_old.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

    def make_action(self, state, action, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        if not test:
            state = torch.from_numpy(state).float().to(device)
        
        state_value = self.model.critic(state)
        action_probs = self.model.actor(state)
        action_distribution = Categorical(action_probs)
        
        if not test:
            action = action_distribution.sample()
            self.actions.append(action)
            self.states.append(state)
            self.logprobs.append(action_distribution.log_prob(action))
            
        action_logprobs = action_distribution.log_prob(action)
        dist_entropy=action_distribution.entropy()
        
        if test:
            return action_logprobs,torch.squeeze(state_value),dist_entropy
        
        if not test:
            return action.item()

       

    def update(self):
        # TODO:
        # discount your saved reward
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list in tensor
        old_states = torch.stack(self.states).to(device).detach()
        old_actions = torch.stack(self.actions).to(device).detach()
        old_logprobs = torch.stack(self.logprobs).to(device).detach()

        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.make_action(old_states, old_actions, test=True)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            

        self.clear_memory()
        # Copy new weights into old policy:
        self.model_old.load_state_dict(self.model.state_dict())

        

    def train(self):
        
        running_reward=0
        avg_length=0
        n_update = 10 
        log_interval = 10
        num_episodes=5000
       
        rewards=[]
        time_step=0
        steps=[]
        max_timesteps=300
        time_step=0
        update_timestep = 2000
        for epoch in range(num_episodes):
            state = self.env.reset()
            for t in range(max_timesteps):
                time_step+=1
                action=self.make_action(state,None,test=False)                
                state, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                
                if time_step % update_timestep == 0:
                    self.update()
                    time_step = 0
                
                running_reward+=reward
                if done:
                    break

            avg_length+=t
                
            if running_reward >50*log_interval:
                self.save('ppo.cpt')
                print("########## Solved! ##########")
                break
                
            if epoch % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = int((running_reward/log_interval))
                rewards.append(running_reward)
                print('Episode {} \t avg length: {} \t reward: {}'.format(epoch, avg_length, running_reward))
                running_reward = 0
                avg_length = 0
                
        plt.plot(rewards)
        plt.ylabel('Moving average  reward')
        plt.xlabel('Step')
        plt.savefig('./ppo50')



            
                
                
                

            
        
            
            