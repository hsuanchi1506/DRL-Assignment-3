# student_agent.py
import gym
import torch
import numpy as np
import cv2
from collections import deque
from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),          nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),          nn.ReLU()
        )
        self.fc_value = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        v = self.fc_value(x)
        a = self.fc_adv(x)
        return v + (a - a.mean(dim=1, keepdim=True))


class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

        self.device = torch.device("cpu")
        self.model  = DuelingDQN(in_channels=4, num_actions=12).to(self.device)
        self.model.load_state_dict(
            torch.load("ckpt/mario_dqn_ep11200.pt", map_location=self.device)
        )
        self.model.eval()

        # 3. frame-skip
        self.skip        = 4
        self._obs_buffer = deque(maxlen=2)   # for max-pool
        self.frames      = deque(maxlen=4)   # for stack

        self.initialized = False
        self.step_count  = 0
        self.last_action = 0


    def reset(self):
        self._obs_buffer.clear()
        self.frames.clear()
        self.initialized = False
        self.step_count  = 0
        self.last_action = 0


    def preprocess(self, obs):
        gray  = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, -1).astype(np.uint8)


    def act(self, observation):
        if not self.initialized:
            self.reset()

        # B. Max-And-Skip  max-pool
        self._obs_buffer.append(observation)
        max_frame = (
            np.max(np.stack(self._obs_buffer), axis=0)
            if len(self._obs_buffer) == 2 else observation
        )

        self.step_count += 1
        need_infer = (not self.initialized) or (self.step_count % self.skip == 0)

        if need_infer:
            proc = self.preprocess(max_frame)

            if not self.initialized:
                for _ in range(4):
                    self.frames.append(proc)
                self.initialized = True
            else:
                self.frames.append(proc)

            # (H,W,C) → (1,C,H,W) and normalize
            state = np.concatenate(self.frames, -1).astype(np.float32) / 255.0
            tensor = (
                torch.from_numpy(state)
                .permute(2, 0, 1)     # C,H,W
                .unsqueeze(0)         # B,C,H,W
                .to(self.device)
            )

            with torch.no_grad():
                q = self.model(tensor)
                self.last_action = q.argmax(dim=1).item()

        return self.last_action
