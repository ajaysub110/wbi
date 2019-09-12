import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms

from modules import ImageEncoder

from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VAE(nn.Module):
    def __init__(self, n_inputs=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), # -1,32,14,14
            nn.Conv2d(32,64,4,2,1), # -1,64,7,7
            nn.Conv2d(64,128,3,2,0), # -1,128,3,3
            nn.Conv2d(128,32,3,1,0), # -1,32,1,1
            Flatten() # -1,32
        )

        self.fc1 = nn.Linear(n_inputs, n_inputs)
        self.fc2 = nn.Linear(n_inputs, n_inputs)
        self.fc3 = nn.Linear(n_inputs,128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,256,4,2,1), # 2,2,256
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

    def bottleneck(self,h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self,x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return z, mu, logvar

    def decode(self,z):
        z = z.view((-1,128,1,1))
        z = self.decoder(z)
        return z 

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

n_epochs = 1
batch_size = 32
SAVE_PATH = './ckpt/image_vae.ckpt'

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)
    KLD = 0.1 * torch.sum(torch.pow(mu, 2) + logvar.exp() -logvar - 1)

    return BCE + KLD, BCE, KLD

def train_image_vae(vae, data_loader):
    optimizer = optim.Adam(vae.parameters(), lr = 0.0001)
    for epoch in range(n_epochs):
        for idx, (images, _) in enumerate(data_loader):
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            bce.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}/{}] loss: {:.3f} bce: {:.3f} kld: {:.3f}".format(epoch+1, 
                                n_epochs,idx, loss.data.item(), bce.data.item(), kld.data.item())
            print(to_print)
        torch.save(vae.state_dict(),SAVE_PATH)

    return vae

def main():

    # Load dataset
    mnist_train_data = datasets.MNIST(
        '/home/ajays/Downloads/',download=True,transform=transforms.ToTensor()
    )
    mnist_test_data = datasets.MNIST('/home/ajays/Downloads/',train=False,download=True)

    train_loader = torch.utils.data.DataLoader(
        mnist_train_data, batch_size = batch_size, shuffle=True
    )

    # Instantiation
    vae = VAE(n_inputs=32)

    # summary(vae,input_size=(1,28,28))
    #  before training
    # o_before, mu, logvar = vae(mnist_train_data[0][0].reshape((1,1,28,28)))
    # plt.imshow(o_before.detach().numpy().reshape((28,28)))
    # plt.show()

    # train
    vae.load_state_dict(torch.load(SAVE_PATH))
    # vae = train_image_vae(vae, train_loader)

    # After training
    example = mnist_train_data[12]
    print(example[1])
    o_after, mu, logvar = vae(example[0].reshape((1,1,28,28)))
    plt.imshow(o_after.detach().numpy().reshape((28,28)))
    plt.show()

if __name__ == '__main__':
    main()