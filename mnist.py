import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms

class EEGEncoder(nn.Module):
    def __init__(self,n_inputs):
        super(Net,self).__init__()
        self.lstm0 = nn.LSTM(self.n_inputs,self.n_inputs)
        self.fc1 = nn.Linear(self.n_inputs,self.n_inputs)
        self.fc2 = nn.Linear(self.n_inputs,self.n_inputs)
        self.fc3 = nn.Linear(self.n_inputs,self.n_inputs)
        self.fc4 = nn.Linear(self.n_inputs,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lstm0(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))

        return x

class VAEDecoder(nn.Module):
    def __init__(self):
        self.dconv3 = nn.ConvTranspose2d(256,128,4,4,0)
        self.dconv4 = nn.ConvTranspose2d(128,64,4,2,1)
        self.dconv5 = nn.ConvTranspose2d(64,32,4,2,1)
        self.dconv6 = nn.ConvTranspose2d(32,1,4,2,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dconv3(x)
        x = self.dconv4(x)
        x = self.dconv5(x)
        x = self.dconv6(x)
        
        return x

class VAEEncoder(nn.Module):
    def __init__(self):
        self.conv0 = nn.Conv2d(1,32,4,2,1)
        self.conv1 = nn.Conv2d(32,32,4,2,1)
        self.conv2 = nn.Conv2d(32,64,4,2,1)
        self.conv3 = nn.Conv2d(64,128,4,2,1)
        self.conv4 = nn.Conv2d(128,256,4,2,1)
        self.fc51 = nn.Linear(256,256)
        self.fc52 = nn.Linear(256,256)

    def encode(self,x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return self.fc51(x), self.fc52(x)

    def sampling(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self,x):
        mu, var = self.encode(x.view(-1))
        z = self.sampling(mu,var)
        return z, mu, var

class VAE(nn.Module):
    def __init__(self):
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def forward(self,x):
        encoded, mu, var = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded, mu, var

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train_VAE(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output, mu, var = model(data)
        loss = loss_function(output, data, mu, var)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def main():
    n_epochs = 10

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])),
        batch_size=16, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])),
        batch_size=16, shuffle=True
    )

    vae_model = VAE()
    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)

    for epoch in range(1, n_epochs+1):
        train_VAE(vae_model,train_loader,optimizer,epoch)

    torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == "__main__":
    main()