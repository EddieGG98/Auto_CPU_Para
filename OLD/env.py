import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import copy

B = 5
DEVICE = "cuda"

LM = SentenceTransformer("all-MiniLM-L12-v2")

class register():

    def __init__(self, data:list, inst:str=""):
        
        self.values = data          # Decimal Contents
        self.inst   = inst          # Record the operations

    def __str__(self) -> str:
        return f"{self.values}\n{self.inst}"

    def is_sorted(self):
        for i in range(len(self.values) - 1):
            if self.values[i] > self.values[i + 1]:
                return False
        return True
    
    def distance(self):
        count = 0
        for i in range(len(self.values) - 1):
            if self.values[i] > self.values[i + 1]:
                count += 1
        return count

    # def get_enc(self):
    #     binary_array = [format((number+2**B)%2**B, f'0{B}b') for number in self.values]
    #     binary_str = "".join(binary_array)
    #     enc = [int(a) for a in binary_str]
    #     return torch.tensor(enc, dtype=torch.float, device=DEVICE)
    
    def get_data_enc(self):
        return torch.tensor(self.values, dtype=torch.float, device=DEVICE)
    
    def get_inst_enc(self): # Size = [384]
        return  torch.tensor(LM.encode(self.inst), dtype=torch.float, device=DEVICE)

def generate_add_task(): # L = 3

    data = [0, np.random.randint(0,2**(B-2)), np.random.randint(0,2**(B-2))]
    ans =  data[1] + data[2]
    return register(data), ans

def generate_sorting_task(L):
    data = sorted([np.random.randint(0, 128) for _ in range(L)], reverse=True)
    return register(data)

def generate_arithmatic_task():
    data = [0, np.random.rand(), np.random.rand()]
    if np.random.rand() > 0.5:
        ans =  data[1] + data[2]
        inst= "Task: Add\n"
    else:
        ans =  data[1] - data[2]
        inst= "Task: Minus\n"
    return register(data, inst), ans