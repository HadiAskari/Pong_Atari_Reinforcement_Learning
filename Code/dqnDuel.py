## dqn.py (Modified as Random Agent)
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.lin = nn.Linear(self.feature_size(), 512)
        self.activation = nn.ReLU()
        
        self.state_value = nn.Sequential(nn.Linear(in_features=512, out_features=256),
           nn.ReLU(),
           nn.Linear(in_features=256, out_features=1),
       )

        self.action_value = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.num_actions),
        )

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        h = self.activation(self.lin(h))
        action_value = self.action_value(h)
        state_value = self.state_value(h)
        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        q = state_value + action_score_centered
        return q

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            ######## YOUR CODE HERE! ########
            # TODO:
            q_value_curr = self.forward(state)
            action = q_value_curr.max(1)[1].data[0] 
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())


def compute_td_loss(model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)), requires_grad = True)
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    
    q_values_current = model(state)
    q_values_next = model(next_state)

    q_value_curr = q_values_current.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = q_values_next.max(1)[0]
    y_value = reward + gamma * next_q_value * (1 - done)
   

    y_value = y_value.detach()
    loss = nn.MSELoss()(y_value, q_value_curr)
    

    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########
        # TODO: Randomly sampling data with specific batch size from the buffer
        sample_batch = random.sample(self.buffer,batch_size)
        state = [samp_bat[0] for samp_bat in sample_batch]
        action = [samp_bat[1] for samp_bat in sample_batch]
        reward = [samp_bat[2] for samp_bat in sample_batch]
        next_state = [samp_bat[3] for samp_bat in sample_batch]
        done = [samp_bat[4] for samp_bat in sample_batch]
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)