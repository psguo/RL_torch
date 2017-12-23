import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
import gym

class Net(nn.Module):
    def __init__(self, input_size, n_l1, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, n_l1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_l1, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class DQN:
    def __init__(self, mem_size, batch_size, n_features, n_actions, gamma, replace_counter, learning_rate, epsilon):
        
        self.n_actions = n_actions
        self.eval_net = Net(n_features, 30, n_actions)
        self.target_net = Net(n_features, 30, n_actions)

        self.n_features = n_features

        self.mem_size = mem_size
        self.mem = np.zeros((self.mem_size, self.n_features * 2 + 2))
        self.mem_counter = 0

        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate) 
        self.criterion = nn.MSELoss()

        self.gamma = gamma

        self.learn_counter = 0

        self.replace_counter = replace_counter
        self.epsilon = epsilon

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s,[a, r], s_))
        index = self.mem_counter % self.mem_size
        self.mem[index, :] = transition
        self.mem_counter += 1

    def choose_action(self,observation):
        observation = Variable(torch.FloatTensor(observation)).view(1,-1)
        if np.random.uniform() < self.epsilon:
            q_eval = self.eval_net.forward(observation)
            action = q_eval.max(1)[1]
            action = action.data[0]
        else:
            action = np.random.randint(self.n_actions)

        return action

    def learn(self):
        # 0. replace target net
        if self.learn_counter % self.replace_counter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        # 1. sampled mem into batch
        sample_size = min(self.mem_counter, self.mem_size)
        sample_index = np.random.choice(sample_size, size=self.batch_size)
        batch_mem = self.mem[sample_index, :]

        # 2. Separate variables
        s = Variable(torch.FloatTensor(batch_mem[:, :self.n_features]))
        a = Variable(torch.LongTensor(batch_mem[:, self.n_features:self.n_features+1]))
        r = Variable(torch.FloatTensor(batch_mem[:, self.n_features+1:self.n_features+2]))
        s_ = Variable(torch.FloatTensor(batch_mem[:, -self.n_features:]))

        # 3. calculate loss
        q_next = self.target_net.forward(s_).detach()
        q_target = r + self.gamma * q_next.max(1)[0].view(-1,1)
        
        q_eval = self.eval_net.forward(s)
        q_eval_wrt_a = q_eval.gather(1, a)

        # 4. back propogation
        self.optimizer.zero_grad()
        loss = self.criterion(q_eval_wrt_a, q_target)
        loss.backward()
        self.optimizer.step()


env = gym.make('CartPole-v0')
env = env.unwrapped

dqn = DQN(mem_size=2000, batch_size=32, n_features=env.observation_space.shape[0], n_actions=env.action_space.n, gamma=0.09, replace_counter=100, learning_rate = 0.01, epsilon = 0.1)

for episode in range(300):
    observation = env.reset()

    while True:
        env.render()
        action = dqn.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # modify the reward
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2

        dqn.store_transition(observation, action, reward, observation_)

        # if dqn.mem_counter > MEMORY_CAPACITY
        
        dqn.learn()

        # swap observation
        observation = observation_

        # break while loop when end of this episode
        if done:
            break