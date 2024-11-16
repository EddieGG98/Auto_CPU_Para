import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from env import task_holder, generate_sorting_task
from models import Bi_MLP

DEVICE = "cuda:1"

class base_agent():

    def __init__(self, L, neighbors, neighbor_encs:list[int], value_net:Bi_MLP, action_net:Bi_MLP):
                
        self.L = L
        self.neighbors = neighbors
        self.neighbors_enc = neighbor_encs

        self.value_net = value_net
        self.target_net = copy.deepcopy(value_net)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()

        self.action_net = action_net
        if self.action_net:
            self.action_target_net = copy.deepcopy(action_net)
            self.action_target_net.load_state_dict(action_net.state_dict())
            self.action_target_net.eval()

    def operate(self, task:task_holder, alpha=1):
        return 0, None
    
    def operate_by_target(self, task:task_holder):
        return 0, None
    
    def operate_random(self, task:task_holder):
        return 0
    
    def choose_next_agent(self, task:task_holder, alpha = 1):
        q_list = []
        for neighbor_id in self.neighbors:
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.value_net(state_enc, self.neighbors_enc[neighbor_id])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        return self.neighbors[idx], q_list[idx]
    
    def choose_next_agent_by_target(self, task:task_holder):
        q_list = []

        for neighbor_id in self.neighbors:
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.target_net(state_enc, self.neighbors_enc[neighbor_id])
            q_list.append(q)

        idx = q_list.index(max(q_list))
        return self.neighbors[idx], q_list[idx]
    
    def update_target(self):
        self.target_net.load_state_dict(self.value_net.state_dict())
        if self.action_net:
            self.action_target_net.load_state_dict(self.action_net.state_dict())


class starter(base_agent):
    def __init__(self, L, neighbors, neighbor_encs, value_net:Bi_MLP, action_net:Bi_MLP):
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)


class alu(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)

        A   = np.random.rand(4, 4)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, task:task_holder, alpha=1):
        q_list = []
        for i in range(4):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_net(state_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        if idx == 0:
            task.rd   = 0
            task.record += f"ALU: output 0\n"
        if idx == 1:
            task.rd   = task.data[task.rs1] + task.data[task.rs2]
            task.record += f"ALU: add\n"
        if idx == 2:
            task.rd   = task.rs1 - task.rs2
            task.record += f"ALU: minus\n"
        if idx == 3:
            task.rd   = int(task.rs1 > task.rs2)
            task.record += f"ALU: cmp\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, task:task_holder):
        q_list = []
        for i in range(4):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(state_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, task:task_holder):
        idx = np.random.randint(0, 4)
        # Operate
        if idx == 0:
            task.rd   = 0
        if idx == 1:
            task.rd   = task.data[task.rs1] + task.data[task.rs2]
        if idx == 2:
            task.rd   = task.rs1 - task.rs2
        if idx == 3:
            task.rd   = int(task.rs1 > task.rs2)
        return idx
    
class pointer1(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)
        A = np.random.rand(L, L)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, task:task_holder, alpha=1):
        q_list = []
        for i in range(self.L):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_net(state_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        task.rs1 = idx
        task.record += f"Pointer 1: move to {idx}\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, task:task_holder):
        q_list = []
        for i in range(self.L):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(state_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, task:task_holder):
        idx = np.random.randint(0, self.L)
        # Operate
        task.rs1 = idx
        return idx
    
class pointer2(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)
        A   = np.random.rand(L, L)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, task:task_holder, alpha=1):
        q_list = []
        for i in range(self.L):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_net(state_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        task.rs2 = idx
        task.record += f"Pointer 2: move to {idx}\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, task:task_holder):
        q_list = []
        for i in range(self.L):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(state_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, task:task_holder):
        idx = np.random.randint(0, self.L)
        # Operate
        task.rs2 = idx
        return idx
    
class writer(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)
        A   = np.random.rand(L, L)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, task:task_holder, alpha=1):
        q_list = []
        for i in range(self.L):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_net(state_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        task.data[idx] = task.rd
        task.record += f"Writer: write {idx}\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, task:task_holder):
        q_list = []
        for i in range(self.L):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(state_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, task:task_holder):
        idx = np.random.randint(0, self.L)
        # Operate
        task.data[idx] = task.rd
        return idx
    
class swaper(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)
        A   = np.random.rand(2, 2)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, task:task_holder, alpha=1):
        q_list = []
        for i in range(2):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_net(state_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        if idx == 0:
            tmp = task.data[task.rs1]
            task.data[task.rs1] = task.data[task.rs2]
            task.data[task.rs2] = tmp
            task.record += f"Swaper: swap\n"
        if idx == 1:
            if task.data[task.rs1] > task.data[task.rs2]:
                tmp = task.data[task.rs1]
                task.data[task.rs1] = task.data[task.rs2]
                task.data[task.rs2] = tmp
            task.record += f"Swaper: compare\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, task:task_holder):
        q_list = []
        for i in range(2):
            state_enc = torch.tensor(task.get_enc(), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(state_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, task:task_holder):
        idx = np.random.randint(0, 2)
        # Operate
        if idx == 0:
            tmp = task.data[task.rs1]
            task.data[task.rs1] = task.data[task.rs2]
            task.data[task.rs2] = tmp
        if idx == 1:
            if task.data[task.rs1] > task.data[task.rs2]:
                tmp = task.data[task.rs1]
                task.data[task.rs1] = task.data[task.rs2]
                task.data[task.rs2] = tmp
        return idx

if __name__ == "__main__":
    import time
    t1 = time.time()
    task, ans = generate_sorting_task()
    print(task)
    task.rs1 = 1
    task.rs2 = 2
    a  = alu (3,[1],[1],Bi_MLP(3,384,4,1,1,1,1).to(DEVICE),Bi_MLP(3,384,4,1,1,1,1).to(DEVICE))
    b  = writer (3,[1],[1],Bi_MLP(3,384,3,1,1,1,1).to(DEVICE),Bi_MLP(3,384,3,1,1,1,1).to(DEVICE))
    t2 = time.time()
    a.operate(task)
    t3 = time.time()
    b.operate(task)
    t4 = time.time()
    print(task)
    print(t2-t1,t3-t2,t4-t3)