import torch.nn as nn
    

# create the model!!

class Megatron3000(nn.Module):
    def __init__(self):
        super(Megatron3000, self).__init__()
        self.Matrix1 = nn.Linear(28**2,64)
        self.Matrix2 = nn.Linear(64,64)
        self.Matrix3 = nn.Linear(64,10)
        self.DO = nn.Dropout(0.1)
        self.R = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.DO(x)
        x = self.Matrix3(x)
        return x.squeeze() # removes the last dimension if it is 1