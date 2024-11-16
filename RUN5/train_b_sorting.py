import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

import torch.multiprocessing as mp
import argparse
from agents import comparer
from env import task_holder, generate_sorting_task
from models import Bi_MLP

mp.set_start_method("spawn", force=True)

DEVICE          = "cuda:0"
L               = 3
B               = 8
EPOCH_NUM       = 1000
EPISODE_NUM     = 10
EPISODE_SIZE    = 100
UPDATE_FREQ     = 1
T               = 20
GAMMA           = 0.5
ETA             = 0.5
REWARD          = 10
LR              = 1e-5
DECAY           = 1e-3
ROLLOUT_T       = 1000
ROLLOUT_D       = 100

def run_episode(episode:int, epoch:int, result_list, agents:list, optimizer:optim.Adam):
    np.random.seed(epoch * EPISODE_NUM + episode)
    torch.manual_seed(epoch * EPISODE_NUM + episode)

    episode_loss = 0
    ACC_count = 0
    optimizer.zero_grad()

    for k in range(EPISODE_SIZE):
        task = generate_sorting_task()
        values, targets, action_values, action_targets, rewards = [], [], [], [], []
        next_idx = 0
        for t in range(1, T + 1):

            current_idx = next_idx
            _, avt = agents[current_idx].operate_by_target(task)
            _, av = agents[current_idx].operate(task, alpha=1.1 * epoch / EPOCH_NUM)
            r = 0

            with torch.no_grad():
                for _ in range(ROLLOUT_T):
                    tmp_tsk = copy.deepcopy(task)
                    for _ in range(ROLLOUT_D):
                        next_i = np.random.randint(1, len(agents) + 1)
                        if next_i == len(agents):
                            if tmp_tsk.is_completed():
                                r += REWARD
                            break
                        agents[next_i].operate_random(tmp_tsk)
            r *= ETA
            r /= ROLLOUT_T

            _, target_value = agents[current_idx].choose_next_agent_by_target(task)
            next_idx, value = agents[current_idx].choose_next_agent(task, alpha=1.1 * epoch / EPOCH_NUM)

            action_values.append(av)
            action_targets.append(avt)
            rewards.append(r)
            values.append(value)
            targets.append(target_value)

            if next_idx == len(agents):
                if task.is_completed():
                    episode_loss += (values[-1] - REWARD) ** 2
                    ACC_count += 1
                else:
                    episode_loss += (values[-1] - 0     ) ** 2
                break

        for m in range(len(values) - 1):
            episode_loss += (values[m] - action_targets[m + 1]) ** 2
            episode_loss += (action_values[m + 1] - GAMMA * targets[m + 1] - rewards[m + 1]) ** 2
    
    episode_loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()
    print(episode,"END","ACC:", ACC_count/EPISODE_SIZE)
    result_list.append((ACC_count, episode_loss.cpu().detach().item()))

def run_test_episode(p, result_list, agents):
    np.random.seed(p)
    torch.manual_seed(p)
    with torch.no_grad():
        success = 0
        for k in range(10):
            task = generate_sorting_task()
            next_idx = 0
            for t in range(1, T + 1):
                current_idx = next_idx
                action_idx, _ = agents[current_idx].operate(task)
                next_idx, _ = agents[current_idx].choose_next_agent(task)
                if next_idx == len(agents):
                    if task.is_completed():
                        success += 1
                        print(task)
                    break
        result_list.append(success)

# Main training loop
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--accname", type=str, default="tmp.png", help="The name of the ACC curve")
    args = parser.parse_args()

    nets_value = [
        Bi_MLP(input_size1=2*L+3, input_size2=2, hidden_dim1=256, hidden_dim2=256, output_dim=1).to(DEVICE),
    ]

    nets_action = [
        Bi_MLP(input_size1=2*L+3, input_size2=L-1, hidden_dim1=256, hidden_dim2=256, output_dim=1).to(DEVICE),
    ]

    for model in nets_action + nets_action:
        model.share_memory()

    A = np.random.rand(2, 2)
    agent_encs, _ = torch.tensor(np.array(np.linalg.qr(A)), dtype=torch.float32, device=DEVICE)

    agents = [
        comparer(L, [0,1], agent_encs, nets_value[0], nets_action[0]),
    ]

    optimizer = optim.Adam(
        [param for net in nets_value + nets_action for param in net.parameters()],
        lr=LR, weight_decay=DECAY
    )

    Train_ACC, Test_ACC = [], []

    for epoch in range(1, EPOCH_NUM + 1):
        print('-' * 30)
        print(f"EPOCH: {epoch}")

        processes = []
        manager = mp.Manager()
        result_list = manager.list([])

        for episode in range(EPISODE_NUM):
            print(episode,"START")
            p = mp.Process(target=run_episode, args=(episode, epoch, result_list, agents, optimizer))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        ACC_count, total_loss = 0, 0
        for acc, loss in result_list:
            ACC_count += acc
            total_loss += loss

        if epoch % UPDATE_FREQ == 0:
            for i in range(len(agents)):
                agents[i].update_target()

        Train_ACC.append        (ACC_count / EPISODE_NUM / EPISODE_SIZE)
        print(f"TRAIN ACC:      {ACC_count / EPISODE_NUM / EPISODE_SIZE}")
        print(f"AVERAGE LOSS:   {total_loss / EPISODE_NUM / EPISODE_SIZE}")


        ### TEST
        NUM_P     = 10
        result_list = manager.list([])
        processes = []
        for k in range(NUM_P):
            p = mp.Process(target=run_test_episode, args=(k, result_list, agents))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        ACC_count = sum(result_list)
        Test_ACC.append         (ACC_count / NUM_P / 10)
        print(f"Test ACC:       {ACC_count / NUM_P / 10}")
        print('-' * 30)

        if epoch % 2 == 0:
            plt.plot(range(1, len(Train_ACC) + 1), Train_ACC, label="Train")
            plt.plot(range(1, len(Test_ACC)  + 1), Test_ACC, label="Test")
            plt.legend()
            plt.savefig(args.accname)
            plt.close()
