import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

from env import register, generate_arithmatic_task
from models import Tri_MLP

DEVICE = "cuda"

class base_agent():

    def __init__(self, L, neighbors, neighbor_encs:list[int], value_net:Tri_MLP, action_net:Tri_MLP):
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

    def operate(self, reg:register, alpha=1):
        return 0, None
    
    def operate_by_target(self, reg:register):
        return 0, None
    
    def operate_random(self, reg:register):
        return 0
    
    def choose_next_agent(self, reg:register, alpha = 1):
        q_list = []
        for neighbor_id in self.neighbors:
            q = self.value_net(reg.get_data_enc(), reg.get_inst_enc(), self.neighbors_enc[neighbor_id])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        return self.neighbors[idx], q_list[idx]
    
    def choose_next_agent_by_target(self, reg:register):
        q_list = []

        for neighbor_id in self.neighbors:
            q = self.target_net(reg.get_data_enc(), reg.get_inst_enc(), self.neighbors_enc[neighbor_id])
            q_list.append(q)

        idx = q_list.index(max(q_list))
        return self.neighbors[idx], q_list[idx]
    
    def update_target(self):
        self.target_net.load_state_dict(self.value_net.state_dict())
        if self.action_net:
            self.action_target_net.load_state_dict(self.action_net.state_dict())


class starter(base_agent):
    def __init__(self, L, neighbors, neighbor_encs, value_net:Tri_MLP, action_net:Tri_MLP):
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)


class swaper(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)

        A   = np.random.rand(3,3)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, reg:register, alpha=1):
        q_list = []
        for i in range(self.L-1):
            q = self.action_net(reg.get_data_enc(), reg.get_inst_enc(), self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        if idx == 0:
            tmp = reg.data[reg.pointers[0]]
            reg.data[reg.pointers[0]] = reg.data[reg.pointers[1]]
            reg.data[reg.pointers[1]] = tmp
            reg.inst += f"SWAPER: Swap data 0 and data 1\n"
        if idx == 1:
            reg.data[reg.pointers[0]] = reg.data[reg.pointers[1]]
            reg.data[reg.pointers[1]] = 0
            reg.inst += f"SWAPER: Move data 1 to data 0\n"
        if idx == 2:
            reg.data[reg.pointers[1]] = reg.data[reg.pointers[0]]
            reg.data[reg.pointers[0]] = 0
            reg.inst += f"SWAPER: Move data 0 to data 1\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, reg:register):
        q_list = []
        for i in range(self.L-1):
            q = self.action_target_net(reg.get_data_enc(), reg.get_inst_enc(), self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, reg:register):
        idx = np.random.randint(0, self.L-1)
        # Operate
        if idx == 0:
            tmp = reg.data[reg.pointers[0]]
            reg.data[reg.pointers[0]] = reg.data[reg.pointers[1]]
            reg.data[reg.pointers[1]] = tmp
            reg.inst += f"SWAPER: Swap data 0 and data 1\n"
        if idx == 1:
            reg.data[reg.pointers[0]] = reg.data[reg.pointers[1]]
            reg.data[reg.pointers[1]] = 0
            reg.inst += f"SWAPER: Move data 1 to data 0\n"
        if idx == 2:
            reg.data[reg.pointers[1]] = reg.data[reg.pointers[0]]
            reg.data[reg.pointers[0]] = 0
            reg.inst += f"SWAPER: Move data 0 to data 1\n"
        return idx

class alu(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)

        A   = np.random.rand(2, 2)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, reg:register, alpha=1):
        q_list = []
        for i in [0, 1]:
            q = self.action_net(reg.get_data_enc(), reg.get_inst_enc(), self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        if idx == 0:
            reg.data[0] = reg.data[reg.pointers[0]] + reg.data[reg.pointers[1]]
            reg.inst += f"ALU: Add\n"
        if idx == 1:
            reg.data[0] = reg.data[reg.pointers[0]] - reg.data[reg.pointers[1]]
            reg.inst += f"ALU: Minus\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, reg:register):
        q_list = []
        for i in [0, 1]:
            q = self.action_target_net(reg.get_data_enc(), reg.get_inst_enc(), self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, reg:register):
        idx = np.random.randint(0, 2)
        # Operate
        if idx == 0:
            reg.data[0] = reg.data[reg.pointers[0]] + reg.data[reg.pointers[1]]
            reg.inst += f"ALU: Add\n"
        if idx == 1:
            reg.data[0] = reg.data[reg.pointers[0]] - reg.data[reg.pointers[1]]
            reg.inst += f"ALU: Minus\n"
        return idx

class pm(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)

        A   = np.random.rand(self.L**2, self.L**2)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, reg:register, alpha=1):
        q_list = []
        for i in range(self.L**2):
            q = self.action_net(reg.get_data_enc(), reg.get_inst_enc(), self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        reg.pointers[0] = idx //self.L
        reg.pointers[1] = idx % self.L
        reg.inst += f"Pointers at {reg.pointers[0]} and {reg.pointers[1]}\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, reg:register):
        q_list = []
        for i in [0, 1]:
            q = self.action_target_net(reg.get_data_enc(), reg.get_inst_enc(), self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, reg:register):
        idx = np.random.randint(0, self.L**2)
        # Operate
        reg.pointers[0] = idx //self.L
        reg.pointers[1] = idx % self.L
        reg.inst += f"Pointers at {reg.pointers[0]} and {reg.pointers[1]}\n"
        return idx


if __name__ == "__main__":
    import time
    t1 = time.time()
    reg, ans = generate_arithmatic_task()
    reg.pointers = [1,2]
    print(reg)
    a = swaper (3,[1],[1],Tri_MLP(4,384,3,1,1,1,1).to(DEVICE),Tri_MLP(4,384,3,1,1,1,1).to(DEVICE))
    b = alu    (3,[1],[1],Tri_MLP(4,384,2,1,1,1,1).to(DEVICE),Tri_MLP(4,384,2,1,1,1,1).to(DEVICE))
    t2 = time.time()
    a.operate(reg)
    t3 = time.time()
    b.operate(reg)
    t4 = time.time()
    print(reg)
    print(t2-t1,t3-t2,t4-t3)