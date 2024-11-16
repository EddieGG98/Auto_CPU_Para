import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

B = 8

class task_holder():

    def __init__(self, data:list, target:list):
        
        self.data   = data          # Decimal Contents
        self.target = target
        self.record = ""

        self.rs1    = 0             # Opt1 reg
        self.rs2    = 0             # Opt2 reg
        self.rd     = 0             # Out  reg

    def __str__(self) -> str:
        return f"{self.data}\n{self.record}"

    def is_completed(self):
        return self.data == self.target
    
    def get_enc(self):              # length = 2L + 3
        return [self.rd, self.rs1, self.rs2]+self.data+self.target


    # def get_enc(self):
    #     binary_array = [format((number+2**B)%2**B, f'0{B}b') for number in self.data]
    #     binary_str = "".join(binary_array)
    #     enc = [int(a) for a in binary_str]
    #     return torch.tensor(enc, dtype=torch.float, device=DEVICE)

def generate_sorting_task():
    data   = [np.random.rand(), np.random.rand(), np.random.rand()]
    target = sorted(data)
    return task_holder(data, target)