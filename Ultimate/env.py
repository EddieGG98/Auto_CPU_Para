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

### register with pointer

class register():
    
    def __init__(self, data:list, inst:str=""):
        
        self.pointers = [0, 0]
        self.data     = data          # Decimal Contents
        self.inst     = inst          # Record the operations

    def __str__(self) -> str:
        return f"{self.data}\n{self.inst}"

    def move_pointer(self, pointer_id, address):
        self.pointers[pointer_id] = address

    def is_sorted(self):
        for i in range(len(self.data) - 1):
            if self.data[i] > self.data[i + 1]:
                return False
        return True
    
    def distance(self):
        count = 0
        for i in range(len(self.data) - 1):
            if self.data[i] > self.data[i + 1]:
                count += 1
        return count
    
    def get_data_enc(self):
        return torch.tensor(self.data, dtype=torch.float, device=DEVICE)
    
    def get_inst_enc(self): # Size = [384]
        return  torch.tensor(LM.encode(self.inst), dtype=torch.float, device=DEVICE)

def generate_arithmatic_task():
    data = [0, np.random.rand(), np.random.rand(), np.random.rand()]

    i = np.random.choice(range(3))

    if i==0:
        ans =  sum(data)
        inst= "Task: Sum\n"
    if i==1:
        ans =  data[2] - data[1]
        inst= "Task: Minus 2 and 1\n"
    if i==2:
        ans =  data[3] - data[2]
        inst= "Task: Minus 3 and 2\n"
    return register(data, inst), ans