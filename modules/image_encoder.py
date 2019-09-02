import torch
import torch.nn as nn 

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ImageEncoder(nn.Module):
    def __init__(self,input_size=28):
        super(ImageEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), # 14,14,32
            nn.Conv2d(32,64,4,2,1), # 7,7,64
            nn.Conv2d(64,128,3,2,0), # 3,3,128
            nn.Conv2d(128,32,3,1,0), # 1,1,32
            Flatten()
        )
    
    def forward(self, x):
        return self.encoder(x)