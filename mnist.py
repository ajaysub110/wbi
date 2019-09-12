import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms

from modules import VAE, ImageEncoder

from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

n_epochs = 5
batch_size = 32
SAVE_PATH = './ckpt/image_vae.ckpt'

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def train_image_vae(vae, data_loader, optimizer):
    for epoch in range(n_epochs):
        for idx, (images, _) in enumerate(data_loader):
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}/{}] loss: {:.3f} bce: {:.3f} kld: {:.3f}".format(epoch+1, 
                                n_epochs,idx, loss.data.item()/batch_size, bce.data.item()/batch_size, 
                                kld.data.item()/batch_size)
            print(to_print)
        torch.save(vae.state_dict(),SAVE_PATH)

def main():
    tfms = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load dataset
    mnist_train_data = datasets.MNIST(
        '/home/ajays/Downloads/',download=True,transform=tfms
    )
    mnist_test_data = datasets.MNIST('/home/ajays/Downloads/',train=False,download=True)

    train_loader = torch.utils.data.DataLoader(
        mnist_train_data, batch_size = batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_test_data, batch_size = batch_size
    )

    # Instantiation
    vae = VAE(n_inputs=32)
    image_encoder = ImageEncoder(input_size=28)

    # Phase 1 - Train image encoder with decoder
    eeg_encoder = vae.encoder
    vae.encoder = image_encoder
    print(vae.encoder)
    optimizer = optim.Adam(vae.parameters(), lr = 1e-3)

    summary(vae,input_size=(1,28,28))
    vae = train_image_vae(vae, train_loader, optimizer)

if __name__ == '__main__':
    main()