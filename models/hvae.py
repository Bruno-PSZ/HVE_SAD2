import torch 
import torch.nn as nn
import torch.nn. functional as F
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(61*61*16, 300)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.reshape((x.shape[0], -1))
        out = F.relu(self.linear1(out))
        return out

class LatetSpace(nn.Module):
    def __init__(self, latent_1_size: int, latent_2_size):
        super().__init__()
        self.latent_1_size = latent_1_size
        self.latent_2_size = latent_2_size

        self.mu1 = nn.Linear(300, latent_1_size)
        self.logvar1 = nn.Linear(300, latent_1_size)

        self.linear2 = nn.Linear(latent_1_size, 64) # rozmiar przejsciowej warstwy pomiedzy z1 i z2

        self.mu2 = nn.Linear(64, latent_2_size)
        self.logvar2 = nn.Linear(64, latent_2_size)

        self.linear3 = nn.Linear(latent_2_size, 64)

        self.mu1_star = nn.Linear(64, latent_1_size)
        self.logvar1_star = nn.Linear(64, latent_1_size)
        

    def forward(self, x):
        mu_1 = torch.sigmoid(self.mu1(x))
        logvar_1 = self.logvar1(x)
        z = Normal(mu_1, torch.exp(logvar_1/2)).rsample()
        
        next_lin = F.relu(self.linear2(z))
        mu_2 = torch.sigmoid(self.mu2(next_lin))
        logvar_2 = self.logvar2(next_lin)
        z2 = Normal(mu_2, torch.exp(logvar_2/2)).rsample()
        
        out = F.relu(self.linear3(z2))
        mu1_star = torch.sigmoid(self.mu1_star(out))
        logvar_1_star = self.logvar1_star(out)
        z1_star = Normal(mu1_star, torch.exp(logvar_1_star/2)).rsample()
        
        return mu_1, logvar_1, mu_2, logvar_2, mu1_star, logvar_1_star, z1_star 
    
class Decoder(nn.Module):
    def __init__(self, latent_1_size):
        super().__init__()
        self.latent_1_size = latent_1_size
        self.linear4 = nn.Linear(latent_1_size, 300)
        self.linear5 = nn.Linear(300, 61*61*16)
        self.conv3 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.conv4 = nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2)

    def unFlatten(self, x):
        return x.reshape((x.shape[0], 16, 61, 61))

    def forward(self, z):
        t = F.relu(self.linear4(z))
        t = F.relu(self.linear5(t))
        t = self.unFlatten(t)
        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))

        t = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)
        return t
    
class HVAE(nn.Module):
    def __init__(self, latent_1_size, latent_2_size, beta=1):
        super().__init__()
        self.latent_1_size = latent_1_size
        self.latent_2_size = latent_2_size
        self.encoder = Encoder()
        self.latent = LatetSpace(latent_1_size, latent_2_size)
        self.decoder = Decoder(latent_1_size)
        self.beta = beta

    def elbo_loss(self, x, pred, mu1, logvar1, mu2, logvar2, mu1_star, logvar_1_star):
        reconstruction_loss = F.mse_loss(pred, x, reduction="sum")

        kld2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=1)
        kld1 = -0.5 * torch.sum(1 + logvar1 - logvar_1_star - logvar1.exp()/logvar_1_star.exp() - ((mu1_star-mu1).pow(2)/logvar_1_star.exp()), dim=1)
        kld_loss = self.beta * (kld1.sum() + kld2.sum())

        return reconstruction_loss, kld_loss

    def forward(self, x):
        last_linear = self.encoder(x)
        mu_1, logvar_1, mu_2, logvar_2, mu1_star, logvar_1_star, z1_star = self.latent(last_linear)
        pred = self.decoder(z1_star)

        return pred, mu_1, logvar_1, mu_2, logvar_2, mu1_star, logvar_1_star
