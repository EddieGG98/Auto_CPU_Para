import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from sentence_transformers import SentenceTransformer
from env import register, generate_arithmatic_task
from models import Tri_MLP

DEVICE = "cuda:4"

class base_agent():

    def __init__(self, L, neighbors, neighbor_encs:list[int], value_net:Tri_MLP, action_net:Tri_MLP):
        
        self.lm = SentenceTransformer("all-MiniLM-L12-v2", device=DEVICE)
        print("LOAD SentenceTransformer")
        
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
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.value_net(data_enc, inst_enc, self.neighbors_enc[neighbor_id])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        return self.neighbors[idx], q_list[idx]
    
    def choose_next_agent_by_target(self, reg:register):
        q_list = []

        for neighbor_id in self.neighbors:
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.target_net(data_enc, inst_enc, self.neighbors_enc[neighbor_id])
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


class alu(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)

        A   = np.random.rand(3, 3)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, reg:register, alpha=1):
        q_list = []
        for i in [0, 1, 2]:
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.action_net(data_enc, inst_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        if idx == 0:
            reg.rd   = reg.rs1 + reg.rs2
            reg.inst += f"ALU: add {0} and {1}\n"
        if idx == 1:
            reg.rd   = reg.rs1 - reg.rs2
            reg.inst += f"ALU: minus {0} and {1}\n"
        if idx == 2:
            reg.rd   = 0
            reg.inst += f"ALU: output 0\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, reg:register):
        q_list = []
        for i in [0, 1, 2]:
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(data_enc, inst_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, reg:register):
        idx = np.random.randint(0, 3)
        # Operate
        if idx == 0:
            reg.rd   = reg.rs1 + reg.rs2
        if idx == 1:
            reg.rd   = reg.rs1 - reg.rs2
        if idx == 2:
            reg.rd   = 0
        return idx
    
class loader1(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)

        A   = np.random.rand(L, L)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, reg:register, alpha=1):
        q_list = []
        for i in range(self.L):
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.action_net(data_enc, inst_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        reg.rs1 = reg.data[idx]
        reg.inst += f"Loader1: load {idx}\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, reg:register):
        q_list = []
        for i in range(self.L):
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(data_enc, inst_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, reg:register):
        idx = np.random.randint(0, self.L)
        # Operate
        reg.rs1 = reg.data[idx]
        return idx
    
class loader2(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)

        A   = np.random.rand(L, L)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, reg:register, alpha=1):
        q_list = []
        for i in range(self.L):
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.action_net(data_enc, inst_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        reg.rs2 = reg.data[idx]
        reg.inst += f"Loader2: load {idx}\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, reg:register):
        q_list = []
        for i in range(self.L):
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(data_enc, inst_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, reg:register):
        idx = np.random.randint(0, self.L)
        # Operate
        reg.rs2 = reg.data[idx]
        return idx
    
class writer(base_agent):

    def __init__(self, L, neighbors, neighbor_encs, value_net:nn.Module, action_net:nn.Module):
        
        super().__init__(L, neighbors, neighbor_encs, value_net, action_net)

        A   = np.random.rand(L, L)
        self.action_enc,_= torch.tensor(np.array(np.linalg.qr(A)),dtype=torch.float32,device=DEVICE)

    def operate(self, reg:register, alpha=1):
        q_list = []
        for i in range(self.L):
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.action_net(data_enc, inst_enc, self.action_enc[i])
            q_list.append(q)
        if np.random.rand() > alpha:
            idx = np.random.randint(0, len(q_list))
        else:
            idx = q_list.index(max(q_list))
        # Operate
        reg.data[idx] = reg.rd
        reg.inst += f"Writer: write {idx}\n"
        return idx, q_list[idx]
    
    def operate_by_target(self, reg:register):
        q_list = []
        for i in range(self.L):
            data_enc = torch.tensor(reg.data, dtype=torch.float, device=DEVICE)
            inst_enc = torch.tensor(self.lm.encode(reg.inst), dtype=torch.float, device=DEVICE)
            q = self.action_target_net(data_enc, inst_enc, self.action_enc[i])
            q_list.append(q)
        idx = q_list.index(max(q_list))
        return idx, q_list[idx]
    
    def operate_random(self, reg:register):
        idx = np.random.randint(0, self.L)
        # Operate
        reg.data[idx] = reg.rd
        return idx

if __name__ == "__main__":
    import time
    t1 = time.time()
    reg, ans = generate_arithmatic_task()
    print(reg)
    a = alu (3,[1],[1],Tri_MLP(3,384,3,1,1,1,1).to(DEVICE),Tri_MLP(3,384,3,1,1,1,1).to(DEVICE))
    b = writer (3,[1],[1],Tri_MLP(3,384,3,1,1,1,1).to(DEVICE),Tri_MLP(3,384,3,1,1,1,1).to(DEVICE))
    t2 = time.time()
    a.operate(reg)
    t3 = time.time()
    b.operate(reg)
    t4 = time.time()
    print(reg)
    print(t2-t1,t3-t2,t4-t3)