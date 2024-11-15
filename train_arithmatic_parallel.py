import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
mp.set_start_method("spawn", force=True)

from agents import starter, alu, loader1, loader2, writer
from env import register, generate_arithmatic_task
from models import Tri_MLP

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--accname", type=str, default="tmp.png", help="The name of the ACC curve")
args = parser.parse_args()

DEVICE          = "cuda"

L               = 3     # Do not change
B               = 8

EPOCH_NUM       = 100
EPISODE_NUM     = 10
EPISODE_SIZE    = 10
UPDATE_FREQ     = 1

T               = 20
GAMMA           = 0.5
ETA             = 0.0
REWARD          = 10

LR              = 1e-3
DECAY           = 1e-3
 
ROLLOUT_T = 10
ROLLOUT_D = 10

def run_episode(episode:int, epoch:int, agents:list[starter], optimizer:optim.Adam):

    np.random.seed(epoch*EPISODE_NUM*EPISODE_SIZE+episode)
    torch.manual_seed(epoch*EPISODE_NUM*EPISODE_SIZE+episode)

    # Initialize episode loss and accuracy count for this specific episode
    episode_loss = 0
    ACC_count = 0

    optimizer.zero_grad()

    for k in range(EPISODE_SIZE):
        reg, ans = generate_arithmatic_task()

        # FORWARDING
        values = []
        targets = []
        action_values = []
        action_targets = []
        rewards = []

        next_idx = 0

        for t in range(1, T + 1):
            current_idx = next_idx
            _, avt = agents[current_idx].operate_by_target(reg)
            _, av = agents[current_idx].operate(reg, alpha=2 * epoch / EPOCH_NUM)

            # Rollout Reward
            r = 0
            with torch.no_grad():
                for _ in range(ROLLOUT_T):
                    tmp_reg = copy.deepcopy(reg)
                    for _ in range(ROLLOUT_D):
                        next_i = np.random.randint(1, len(agents) + 1)
                        if next_i == len(agents):
                            if tmp_reg.data == ans:
                                r += REWARD
                            break
                        agents[next_i].operate_random(tmp_reg)
            r *= ETA
            r /= ROLLOUT_T

            _, target_value = agents[current_idx].choose_next_agent_by_target(reg)
            next_idx, value = agents[current_idx].choose_next_agent(reg, alpha=2 * epoch / EPOCH_NUM)

            action_values.append(av)
            action_targets.append(avt)

            rewards.append(r)

            values.append(value)
            targets.append(target_value)

            if next_idx == len(agents):
                if reg.data[0] == ans:
                    episode_loss += (values[-1] - REWARD) ** 2
                    ACC_count += 1
                    # print(reg)
                else:
                    episode_loss += (values[-1] - 0) ** 2
                break

        # ADD TO EPISODE LOSS
        for m in range(len(values) - 1):
            episode_loss += (values[m] - action_targets[m + 1]) ** 2
            episode_loss += (action_values[m + 1] - GAMMA * targets[m + 1] - rewards[m + 1]) ** 2
    
    # print(f"{episode} In: ",agents[0].value_net.state_dict()['outnet.fcs.0.bias'][0:3])
    # print(ACC_count)
    episode_loss.backward()
    optimizer.step()
    # print(f"{episode} Out:",agents[0].value_net.state_dict()['outnet.fcs.0.bias'][0:3])
    return ACC_count, episode_loss.cpu().detach().item()

# Main training loop
if __name__ == "__main__":

    nets_value = [
    Tri_MLP(input_size1 = L,
            input_size2 = 384,
            input_size3 = 6,
            hidden_dim1 = 256,
            hidden_dim2 = 256,
            hidden_dim3 = 256,
            output_dim  = 1          ).to(DEVICE) for _ in range(5)]

    nets_action = [
        Tri_MLP(input_size1 = L,
                input_size2 = 384,
                input_size3 = L,
                hidden_dim1 = 256,
                hidden_dim2 = 256,
                hidden_dim3 = 256,
                output_dim  = 1          ).to(DEVICE),
        Tri_MLP(input_size1 = L,
                input_size2 = 384,
                input_size3 = L,
                hidden_dim1 = 256,
                hidden_dim2 = 256,
                hidden_dim3 = 256,
                output_dim  = 1          ).to(DEVICE),
        Tri_MLP(input_size1 = L,
                input_size2 = 384,
                input_size3 = L,
                hidden_dim1 = 256,
                hidden_dim2 = 256,
                hidden_dim3 = 256,
                output_dim  = 1          ).to(DEVICE),
        Tri_MLP(input_size1 = L,
                input_size2 = 384,
                input_size3 = 3,
                hidden_dim1 = 256,
                hidden_dim2 = 256,
                hidden_dim3 = 256,
                output_dim  = 1          ).to(DEVICE)]

    A   = np.random.rand(6, 6)
    agent_encs ,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    agents =  [starter(L, [1], agent_encs, nets_value[0], None          )] \
            + [loader1(L, [2], agent_encs, nets_value[1], nets_action[0])] \
            + [loader2(L, [4], agent_encs, nets_value[2], nets_action[1])] \
            + [writer (L, [5], agent_encs, nets_value[3], nets_action[2])] \
            + [alu    (L, [3], agent_encs, nets_value[4], nets_action[3])]

    optimizer = optim.Adam([param for net in nets_value+nets_action for param in net.parameters()],
                            lr = LR, 
                            weight_decay = DECAY)

    Train_ACC = []
    Test_ACC  = []

    for epoch in range(1, EPOCH_NUM + 1):
        print(f"EPOCH: {epoch}")

        ### Train
        ACC_count = 0
        total_loss = 0

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_episode, episode, epoch, agents, optimizer)
                    for episode in range(EPISODE_NUM)]
            for future in as_completed(futures):
                acc, loss = future.result()
                ACC_count += acc
                total_loss += loss

        # print("Finish:",nets_value[0].state_dict()['outnet.fcs.0.bias'][0:3])
        if epoch % UPDATE_FREQ == 0:
            for i in range(len(agents)):
                agents[i].update_target()

        Train_ACC.append(ACC_count / EPISODE_NUM / EPISODE_SIZE)
        print()
        print('-' * 30)
        print(f"EPOCH {epoch} TRAIN ACC: {ACC_count / EPISODE_NUM / EPISODE_SIZE}")
        print(f"AVERAGE LOSS: {total_loss / EPISODE_NUM}")
        print('-' * 30)
        print()
        print("TEST:")

        ## Test

        np.random.seed(0)
        torch.manual_seed(0)

        ACC_count = 0

        for _ in range(10): 
            reg, ans = generate_arithmatic_task()

            next_idx = 0
            # print("-------")
            # print("In  REG:",reg)
            for t in range(1, T+1):
                current_idx = next_idx
                action_idx,_ = agents[current_idx].operate(reg)
                next_idx,  _ = agents[current_idx].choose_next_agent(reg)
                if next_idx == len(agents):
                    if ans == reg.data[0]:
                        ACC_count += 1
                        print("Success",reg)
                    break
            # print("Out REG:",reg)
            # print("-------")
        
        Test_ACC.append(ACC_count/10) # OR reward_count/len(testset)
        print()
        print('-'*30)
        print(f"EPOCH {epoch} TEST  ACC: {ACC_count/10}")
        print('-'*30)
        print()

        if epoch % 10 == 0:
            plt.plot(range(1, len(Train_ACC)+1), Train_ACC, label= "Train")
            plt.plot(range(1, len(Test_ACC) +1), Test_ACC,  label= "Test " )
            plt.legend()
            plt.savefig(args.accname)
            plt.close()