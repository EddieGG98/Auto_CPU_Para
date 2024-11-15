import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

from agents import starter, swaper, alu
from env import register, generate_arithmatic_task
from models import Tri_MLP

L = 4

ETA             = 1
REWARD          = 10
ROLLOUT_T       = 10000
ROLLOUT_D       = 5

nets_value = [
    Tri_MLP(input_size1 = L,
            input_size2 = 384,
            input_size3 = 4,
            hidden_dim1 = 256,
            hidden_dim2 = 256,
            hidden_dim3 = 256,
            output_dim  = 1          ).to("cuda") for _ in range(3)]

nets_action = [
    Tri_MLP(input_size1 = L,
            input_size2 = 384,
            input_size3 = L-1,
            hidden_dim1 = 256,
            hidden_dim2 = 256,
            hidden_dim3 = 256,
            output_dim  = 1          ).to("cuda"),
    Tri_MLP(input_size1 = L,
            input_size2 = 384,
            input_size3 = 2,
            hidden_dim1 = 256,
            hidden_dim2 = 256,
            hidden_dim3 = 256,
            output_dim  = 1          ).to("cuda")]

A   = np.random.rand(4, 4)
agent_encs ,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device="cuda")

agents =  [starter(L, [1,2  ], agent_encs, nets_value[0], None          )] \
        + [swaper (L, [1,2,3], agent_encs, nets_value[1], nets_action[0])] \
        + [alu    (L, [1,2,3], agent_encs, nets_value[2], nets_action[1])]

reg, ans = generate_arithmatic_task()

print(reg)

# Rollout Reward
r = 0
with torch.no_grad():
    for _ in range(ROLLOUT_T):
        tmp_reg = copy.deepcopy(reg)
        for _ in range(ROLLOUT_D):
            next_i = np.random.randint(1,len(agents)+1)
            if next_i == len(agents):
                if tmp_reg.values[0] == ans:
                    r += REWARD*ETA
                break
            agents[next_i].operate_random(tmp_reg)
r /= ROLLOUT_T

print(r)

reg.values[0] = reg.values[2] + reg.values[1]

print(reg)

# Rollout Reward
r = 0
with torch.no_grad():
    for _ in range(ROLLOUT_T):
        tmp_reg = copy.deepcopy(reg)
        for _ in range(ROLLOUT_D):
            next_i = np.random.randint(1,len(agents)+1)
            if next_i == len(agents):
                if tmp_reg.values[0] == ans:
                    r += REWARD*ETA
                break
            agents[next_i].operate_random(tmp_reg)
r /= ROLLOUT_T

print(r)