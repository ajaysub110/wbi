import torch
import torch.nn as nn 

class VAE(nn.Module):
    def __init__(self, n_inputs=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.LSTM(n_inputs, n_inputs, 1),
            nn.Linear(n_inputs,n_inputs),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(n_inputs, n_inputs)
        self.fc2 = nn.Linear(n_inputs, n_inputs)
        self.fc3 = nn.Linear(n_inputs, n_inputs)
        self.fc4 = nn.Linear(n_inputs,1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1,256,4,2,1), # 2,2,256
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1), # 4,4,128
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), # 8,8,64
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), # 16,16,32
            nn.ReLU(),
            nn.ConvTranspose2d(32,1,2,2,2), # 28, 28, 1
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp 
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self,x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar 

    def decode(self,z):
        z = self.fc3(z)
        z = self.fc4(z)
        z = z.view((-1,1,1,1))
        z = self.decoder(z)
        return z 

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar