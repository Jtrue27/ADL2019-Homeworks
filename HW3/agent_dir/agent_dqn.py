import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import ipdb
from  collections import namedtuple
from agent_dir.agent import Agent
from environment import Environment
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        
        self.head = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.fc(x.view(x.size(0), -1))
        x = self.lrelu(x)
        q = self.head(x)
        return q

class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n
        # TODO:
        # Initialize your replay buffer
        # 2000 is memory capacity
        self.memory = ReplayMemory(10000)
        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn')
        
        # discounted reward
        self.GAMMA = 0.99
        # greedy
        self.EPSILON=0.99
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 100 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        # self.num_timesteps = 30000 # total training steps
        self.display_freq = 100 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network
        
        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)
       

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        # TODO:
        # At first, you decide whether you want to explore the environemnt
        if test==True:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state

        if np.random.uniform() < self.EPSILON:   # greedy
            
            actions_value=self.online_net.forward(state)
            actions_value=torch.sort(actions_value,descending=True)
            action=actions_value[1]
        # TODO:
        # if explore, you randomly samples one action
        # else, use your model to predict action
        else:   
            actions_value=torch.randn(1,7) 
            actions_value=torch.sort(actions_value,descending=True)
            action=actions_value[1].cuda() if use_cuda else actions_value[1]
        if test==True:
            action=action[0,0].data.item()


      
        return action

    

    def update(self):
        # TODO:
        # To update model, we sample some stored experiences as training examples.
        # sample batch transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        
        # 切狀態
        b_s=batch[0][0]
        for i in range(1,len(batch[0])):
            b_s = torch.cat((b_s,batch[0][i]),0)
     
        
        b_a =torch.tensor([r for r in batch[1]])
        b_a=b_a.cuda() if use_cuda else b_a

        b_s_ = [state for state in batch[2]]
        b_s_=torch.stack(b_s_,0)
        b_s_=torch.squeeze(b_s_)

        b_r =torch.tensor([r for r in batch[3]])
        b_r=b_r.cuda() if use_cuda else b_r
       
        
        
        # TODO:
        # Compute Q(s_t, a) with your model.
        b_a=torch.unsqueeze(b_a,1)
        q_online = self.online_net(b_s).gather(1,b_a)
        # q_online= q_online.max(1)[0].view(self.batch_size, 1) 
       
        with torch.no_grad():
            # TODO:
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            
            q_next = self.target_net(b_s_)     # detach from graph, don't backpropagate

        # TODO:
        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        # 切成直的
        b_r=torch.unsqueeze(b_r,1)
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        # TODO:
        # Compute temporal difference loss
        loss=F.smooth_l1_loss(q_online,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0 
        n_reward=[]
        n_step=[]
        while(True):
            
            state = self.env.reset()
            # self.env.render()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            
            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)

                
                next_state, reward, done, _ = self.env.step(action[0, 0].data.item())
                action=action[0, 0].data.item()
                total_reward += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None
                    break
                    

                # TODO:
                # store the transition in memory
                
                self.memory.push(state,action,next_state,reward)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                # step 大於 10000 開始更新網路 且 step 是 train 的 頻率開始更新
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network 更新target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn')

                self.steps += 1
                # store rewards 
                
            


            # 10 episode顯示
            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                # add plot graph data
                n_reward.append(total_reward / self.display_freq)
                n_step.append(self.steps)

                total_reward = 0
                
           


            episodes_done_num += 1
            
            if self.steps > self.num_timesteps:
                plt.plot(n_step,n_reward)
                plt.ylabel('Moving average n=100 reward')
                plt.xlabel('Step')
                plt.savefig('./epsilon088')
                break
                
        self.save('dqn')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)