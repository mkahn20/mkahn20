import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, info, mode = 'global'):
        _, num_genre, num_contents, _, _ = info
        
        if mode ==  'global':
            input_size = num_contents
            num_categories = num_contents
            
        elif mode == 'split':
            input_size = num_contents//num_genre
            num_categories = num_contents//num_genre
            
        hidden_size = 256
        latent_size = 32
        temperature = 15

        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_categories = num_categories
        self.temperature = temperature

        # Encoder
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_categories)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mean)
        else:
            z = mean
        return z

    def gumbel_softmax(self, logits):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        softmax = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        return softmax

    def decode(self, z):
        h = F.relu(self.fc3(z))
        logits = self.fc4(h)
        probas = self.gumbel_softmax(logits)
        return probas

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mean, logvar

    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.latent_size).to("cuda:0")
        probas = self.decode(z)
        return probas
        