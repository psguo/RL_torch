import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
import gym
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, input_size, n_l1, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, n_l1)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc1.bias.data.zero_()
        self.fc1.bias.data += 0.1
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(n_l1, num_classes)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.zero_()
        self.fc2.bias.data += 0.1

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out

class PG:
    def __init__(self, n_features, n_actions, learning_rate, gamma):
        self.net = Net(n_features, 10, n_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate) 
        self.criterion = nn.CrossEntropyLoss(reduce=False)

        self.gamma = gamma

        self.n_actions = n_actions

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def choose_action(self,observation):
        observation = Variable(torch.FloatTensor(observation)).view(1,-1)
        prob_weights = self.net.forward(observation)

        softmax = nn.Softmax(dim=1)
        prob_weights = softmax(prob_weights)
        prob_weights.squeeze_()

        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights.data.numpy())
        
        return action

    def learn(self):
        # 1. get reward
        discounted_ep_rs_norm = Variable(torch.FloatTensor(self._discount_and_norm_rewards()))

        # 2. calculate loss
        ep_obs_var = Variable(torch.FloatTensor(self.ep_obs))
        all_act_prob = self.net.forward(ep_obs_var)

        ep_as_var = Variable(torch.LongTensor(np.array(self.ep_as)))

        loss = self.criterion(all_act_prob, ep_as_var)
        loss = torch.mean(loss * discounted_ep_rs_norm)

        # 3. back propogation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. clear memory
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs    


env = gym.make('MountainCar-v0')
env = env.unwrapped

pg = PG(n_features = env.observation_space.shape[0], n_actions = env.action_space.n, learning_rate = 0.02, gamma = 0.995)

for episode in range(1000):
    observation = env.reset()
    while True:
        if episode == 999:
            env.render()
        action = pg.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        pg.store_transition(observation, action, reward)

        observation = observation_

        if done:
            # calculate running reward
            ep_rs_sum = sum(pg.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            print("episode:", episode, "  reward:", int(running_reward))

            pg.learn()
            break