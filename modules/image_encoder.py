import torch
import torch.nn as nn 

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ImageEncoder(nn.Module):
    def __init__(self,input_size=28):
        super(ImageEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), # -1,32,14,14
            nn.Conv2d(32,64,4,2,1), # -1,64,7,7
            nn.Conv2d(64,128,3,2,0), # -1,128,3,3
            nn.Conv2d(128,32,3,1,0), # -1,32,1,1
            Flatten() # -1,32
        )
    
    def forward(self, x):
        return self.encoder(x)