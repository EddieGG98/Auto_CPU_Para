import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

B = 8

class register():

    def __init__(self, data:list, inst:str=""):
        
        self.data = data          # Decimal Contents
        self.inst   = inst          # Record the operations

        self.rs1    = 0             # Opt1 reg
        self.rs2    = 0             # Opt2 reg
        self.rd     = 0             # Out  reg

    def __str__(self) -> str:
        return f"{self.data}\n{self.inst}"

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

    # def get_enc(self):
    #     binary_array = [format((number+2**B)%2**B, f'0{B}b') for number in self.data]
    #     binary_str = "".join(binary_array)
    #     enc = [int(a) for a in binary_str]
    #     return torch.tensor(enc, dtype=torch.float, device=DEVICE)

def generate_arithmatic_task():
    data = [0, np.random.rand(), np.random.rand()]
    if np.random.rand() > 0.5:
        ans =  data[1] + data[2]
        inst= "Task: Add\n"
    else:
        ans =  data[1] - data[2]
        inst= "Task: Minus\n"
    return register(data, inst), ans