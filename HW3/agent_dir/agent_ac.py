import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Categorical
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

class Actor(nn.Module):
    def __init__(self,input_size,hidden_size,action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)
        

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

class Value(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def roll_out(actor_network,task,sample_nums,value_network,init_state):
    #task.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for j in range(sample_nums):
        states.append(state)
        log_softmax_action = actor_network(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)
        # import ipdb; ipdb.set_trace()

        action = np.random.choice(2,p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(4)]
        next_state,reward,done,_ = task.step(action)
        #fix_reward = -10 if done else 1
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = task.reset()
            break
    if not is_done:
        final_r = value_network(Variable(torch.Tensor([final_state]))).cpu().data.numpy()

    return states,actions,rewards,final_r,state

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
    

class AgentAC(Agent):
    def __init__(self, env, args):
        self.env = env
        self.state_dim=self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        
        self.value_net = Value(self.state_dim,40,1)
        self.value_net = self.value_net.cuda() if use_cuda else self.value_net
        self.actor_net = Actor(self.state_dim,40, self.num_actions)
        self.actor_net = self.actor_net.cuda() if use_cuda else self.actor_net

    
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
        self.value_network_optim = torch.optim.Adam(self.value_net.parameters(),lr=0.01)
        self.actor_network_optim = torch.optim.Adam(self.actor_net.parameters(),lr = 0.01)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration


    def save(self, save_path):
        pass

    def load(self, load_path):
        pass

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass


    def train(self):
        steps =[]
        task_episodes =[]
        test_results =[]
        init_state = self.env.reset()
        for step in range(120):
            states,actions,rewards,final_r,current_state = roll_out(self.actor_net,self.env,30,self.value_net,init_state)
            init_state = current_state
            actions_var = Variable(torch.Tensor(actions).view(-1,2))
            states_var = Variable(torch.Tensor(states).view(-1,self.state_dim))

            # train actor network
            self.actor_network_optim.zero_grad()
            log_softmax_actions = self.actor_net(states_var)
            vs = self.value_net(states_var).detach()
            # calculate qs
            qs = Variable(torch.Tensor(discount_reward(rewards,0.99,final_r)))

            advantages = qs - vs
            import ipdb; ipdb.set_trace()
            actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages)
            actor_network_loss.backward()
            torch.nn.utils.clip_grad_norm(self.actor_net.parameters(),0.5)
            self.actor_network_optim.step()

            # train value network
            self.value_network_optim.zero_grad()
            target_values = qs
            values = self.value_net(states_var)
            criterion = nn.MSELoss()
            value_network_loss = criterion(values,target_values)
            value_network_loss.backward()
            torch.nn.utils.clip_grad_norm(self.value_net.parameters(),0.5)
            self.value_network_optim.step()
                
            # Testing
            if (step + 1) %1== 0:
                    result = 0
                    
                    for test_epi in range(10):
                        state = self.env.reset()
                        for test_step in range(200):
                            softmax_action = torch.exp(self.actor_net(Variable(torch.Tensor([state]))))
                            #print(softmax_action.data)
                            action = np.argmax(softmax_action.data.numpy()[0])
                            next_state,reward,done,_ = self.env.step(action)
                            result += reward
                            state = next_state
                            if done:
                                break
                    print("step:",step+1,"test result:",result/10.0)
                    steps.append(step+1)
                    test_results.append(result/10)

        plt.plot(steps,test_results)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()



            
        
            
            