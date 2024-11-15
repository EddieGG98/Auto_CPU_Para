import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

from agents import starter, swaper, alu
from env import register, generate_arithmatic_task
from models import Tri_MLP

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--accname", type=str, default="tmp.png", help="The name of the ACC curve")
args = parser.parse_args()
print(args.accname)


L               = 3     # Do not change
B               = 8

EPOCH_NUM       = 100
EPISODE_NUM     = 100
EPISODE_SIZE    = 10
UPDATE_FREQ     = 1

T               = 20
GAMMA           = 0.5
ETA             = 1
REWARD          = 10

LR              = 1e-5 
DECAY           = 1e-3
 
ROLLOUT_T = 1000
ROLLOUT_D = 5

np.random.seed(0)
torch.manual_seed(0)

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

optimizer = optim.Adam([param for net in nets_value+nets_action for param in net.parameters()],
                        lr = LR, 
                        weight_decay = DECAY)

Train_ACC = []
Test_ACC  = []

for epoch in range(1, EPOCH_NUM+1):
    print(f"EPOCH: {epoch}")

    ### Train
    ACC_count = 0

    for episode in range(EPISODE_NUM):
        print(f"EPISODE: {episode+1}")
        optimizer.zero_grad()
        episode_loss = 0

        for k in range(EPISODE_SIZE):
            
            reg, ans = generate_arithmatic_task()

            ### FORWARDING
            values    = []
            targets   = []
            action_values   = []
            action_targets  = []
            rewards         = []

            distance_records = []

            next_idx:int = 0

            for t in range(1, T+1):
                current_idx = next_idx
                _, avt = agents[current_idx].operate_by_target(reg)
                _, av  = agents[current_idx].operate(reg, alpha=2*epoch/EPOCH_NUM)

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

                _, target_value = agents[current_idx].choose_next_agent_by_target(reg)
                next_idx, value = agents[current_idx].choose_next_agent(reg, alpha=2*epoch/EPOCH_NUM)


                action_values.append(av)
                action_targets.append(avt)

                rewards.append(r)

                values.append(value)
                targets.append(target_value)

                distance_records.append(abs(ans-reg.values[0]))

                if next_idx == len(agents):
                    if reg.values[0] == ans:
                        episode_loss += (values[-1] - REWARD)**2
                        ACC_count += 1
                        print("Success")
                    else:
                        episode_loss += (values[-1] - 0     )**2
                    break

            ### ADD TO EPISODE LOSS
            for m in range(len(values)-1):
                episode_loss += (values[m]-action_targets[m+1])**2
                episode_loss += (action_values[m+1]-GAMMA*targets[m+1]-rewards[m+1])**2

        episode_loss.backward()
        optimizer.step()

    if epoch % UPDATE_FREQ == 0:
        for i in range(len(agents)):
            agents[i].update_target()


    Train_ACC.append(ACC_count/EPISODE_NUM/EPISODE_SIZE) # OR reward_count/len(testset)
    print()
    print('-'*30)
    print(f"EPOCH {epoch} TRAIN ACC: {ACC_count/EPISODE_NUM/EPISODE_SIZE}")
    print('-'*30)
    print()
    print("TEST:")

    ## Test

    ACC_count = 0

    for _ in range(100): 
        reg, ans = generate_arithmatic_task()

        next_idx = 0
        print("-------")
        print("In  REG:",reg)
        for t in range(1, T+1):
            current_idx = next_idx
            action_idx,_ = agents[current_idx].operate(reg)
            next_idx,  _ = agents[current_idx].choose_next_agent(reg)
            if next_idx == len(agents):
                if ans == reg.values[0]:
                    ACC_count += 1
                break
        print("Out REG:",reg)
        print("-------")
    
    Test_ACC.append(ACC_count/100) # OR reward_count/len(testset)
    print()
    print('-'*30)
    print(f"EPOCH {epoch} TEST  ACC: {ACC_count/100}")
    print('-'*30)
    print()

    if epoch % 10 == 0:
        plt.plot(range(1, len(Train_ACC)+1), Train_ACC, label= "Train")
        plt.plot(range(1, len(Test_ACC) +1), Test_ACC,  label= "Test " )
        plt.legend()
        plt.savefig(args.accname)
        plt.close()