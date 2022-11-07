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
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
    state, action, reward, next_state, done = replay_buffer.sample(batch_size,0.4)

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
    def __init__(self, capacity, alpha):

        self.capacity = capacity
        self.alpha = alpha
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1

        self.data = {
            'state': np.zeros(shape=(capacity, 1, 84, 84), dtype=np.uint8),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_state': np.zeros(shape=(capacity, 1, 84, 84), dtype=np.uint8),
            'done': np.zeros(shape=capacity, dtype=np.bool)
       }

        self.next_idx = 0
        self.size = 0


    def add(self, state, action, reward, next_state, done):
        idx = self.next_idx
        self.data['state'][idx] = state
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_state'][idx] = next_state
        self.data['done'][idx] = done
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])
    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]


    def _min(self): 
        return self.priority_min[1]
   
    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size, beta):
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight

        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        
        return (samples['states'],samples['action'],samples['reward'],samples['next_states'],samples['done'])

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
    
    def is_full(self):
        return self.capacity == self.size

    def __len__(self):
        return self.size

        
    # def sample(self, batch_size):
    #     ######## YOUR CODE HERE! ########
    #     # TODO: Randomly sampling data with specific batch size from the buffer
    #     sample_batch = random.sample(self.buffer,batch_size)
    #     state = [samp_bat[0] for samp_bat in sample_batch]
    #     action = [samp_bat[1] for samp_bat in sample_batch]
    #     reward = [samp_bat[2] for samp_bat in sample_batch]
    #     next_state = [samp_bat[3] for samp_bat in sample_batch]
    #     done = [samp_bat[4] for samp_bat in sample_batch]
    #     return np.concatenate(state), action, reward, np.concatenate(next_state), done

