import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
 
    def __init__(self,n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)     

    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.out(x)                 
        return x
 