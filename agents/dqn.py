# Define environment 
import gymnasium as gym
env = gym.make("ALE/Breakout-v5", render_mode = 'human')
import torch as t
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import torch.optim as optim

# A1 Environment
# A1.1 Install packages
# A1.2 Create environment
# A1.3 Understanding action space
# A1.4 Taking random actions
# A1.5 Rendering actions
# A1.6 Understanding observation space


# A3.2 Define a dataframe for experience replay buffer
columns = ["State", "Action", "Reward", "Done"]
rb = pd.DataFrame(columns=columns)

num_actions = 4

learning_rate = 0.00025
alpha = 0.9
eps = 1e-08
weight_decay = 0
momentum = 0
centered = False

# A2 Dataset pre-processing
def frame_preprocessing(frame):
    # A2 Dataset preprocessing
    # A2.1 Convert RGB to gray scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A2.2 Downsampling to 110x84
    new_width = 110
    new_height = 84

    downsampled_frame = cv2.resize(gray_frame, (110, 84))


    # A2.3 Cropping the image
    state = downsampled_frame[:,13:97] # 110-84 
    
    return state

# A3 Define replay buffer
def experience_replay(s, a, r, d):
    # A3.3 Add state, action, reward and done to dataframe
    rb.loc[len(rb)] = [state, action, reward, done]
    return rb

def data_preprocessing(state):
    frame = np.asarray(state[0])
    state = frame_preprocessing(frame)
    return state

def mini_batch(rb):
    return 0

def state_sequence(rb):
    seq = rb.tail(4)
    rb = rb.iloc[:-4]
    return seq, rb

done = False
# while not done:

replay_buffer_size = 1000

# Define agent
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()

        self.fc3 = nn.Linear(self.fc_input_shape(input_shape),512)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(256, num_actions)

    def fc_input_shape(self, input_shape):

        dummy_input = t.zeros(1, *input_shape)
        dummy_output = self.relu3(self.conv3(self.relu2(self.conv2(self.relu1(self.conv1(dummy_input))))))

        return int(dummy_output.view(-1).size()[0])

    def forward(self, x):

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)

        return x

    def q_value():
        return 0

    def loss_function():
        return 0

for i in range(0, replay_buffer_size):
    state = env.reset()
    state = data_preprocessing(state)
    print(state.shape)

    # Random action
    action = env.action_space.sample()
    ##  print(action) 
    # action_space = Discrete(4) 
    # 0 NOOP (No action)
    # 1 FIRE
    # 2 RIGHT
    # 3 LEFT
    # state, action, reward, done = 
    # A3.1 Define state, action, reward

    state, reward, done, truncate, info = env.step(action) # truncate is when the episode length exceeds the set timelimit
    replay_buffer = experience_replay(state, action, reward, done)

    # frame = env.render()
    # print(env.observation_space)

    # Display frame
    # cv2.imshow("Frame", frame)
    # cv2.imshow("State", state)
    # cv2.waitKey(0)

print(replay_buffer)

rb = replay_buffer

# Create Q-Networks
q = DQN(state.shape, num_actions)
q_trans = q.copy()


optimizer = optim.RMSprop(q.parameters(), lr=learning_rate, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)

def train():
    mini_batch = 32

"""

while not done:
    state = env.reset()
    seq, rb = state_sequence(rb)
    state = seq["State"]
    dqn = DQN(state.shape, num_actions )
    action = DQN.forward(state)
    env.step(action)

env.close()

"""

# Define DQN network

# Define experience replay

# Define data pre-processing
