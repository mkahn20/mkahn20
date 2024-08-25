import torch
import random
from torch import optim
from VAE import *
from Sampling import *


class Client():
    def __init__(self, rank, data, local_epoch, ES):
        # seed
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        self.rank = rank
        self.local_epoch = local_epoch
        self.ES = ES
        self.model = VAE(ES.info, 'global').to(ES.device)
        self.ws = None
        self.data = data
        self.num_user, self.num_genre, self.num_contents, self.num_request, self.num_zone = ES.info
        gr = [0 for _ in range(self.num_genre)]
        for i in self.data:
            for idx, j in enumerate(i):
                if j != 0:
                    g = idx//(self.num_contents//self.num_genre)
                    gr[g] += j
        gr = [torch.tensor(gr).topk(self.num_genre)[0].tolist(), torch.tensor(gr).topk(self.num_genre)[1].tolist()]
        self.rk = gr
    
    def load_global_model(self):
        model = VAE(self.ES.info, 'global').to(self.ES.device)
        model.load_state_dict(self.ES.model.state_dict()) 
        return model

    def train(self, model, batch, num_epochs = 10):
        # update local model
        optimizer = optim.Adam(model.parameters(), lr=1e-3,  amsgrad=True)
        #num_epochs = 10
        #batch = self.data
        for _ in range(num_epochs):
            model.train()
            train_loss = 0.0
            batch = torch.tensor(batch, dtype=torch.float32).to(self.ES.device)
            inputs = batch
            optimizer.zero_grad()
            reconstructed_x, mean, logvar = model(inputs)
            
            # Reconstruction loss
            reconstruction_loss = F.binary_cross_entropy(reconstructed_x, inputs, reduction='sum')

            # KL divergence loss
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

            # Total loss
            loss = reconstruction_loss + kl_divergence_loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        # Print progress
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))

    def set_data(self):
        dist = self.ES.dist
        data  = []
        rk    = self.rk[1]
        rt    = self.rk[0]
        
        for idx, g in enumerate(rk):
            dt = inverse_transform_sampling(cdf = CDF(dist[g]),
                        n_samples = int(rt[idx]),
                        left = 0, right = self.num_contents//self.num_genre - 1, seed = self.ES.ESe)
            for c in dt:
                ct = c + (self.num_contents//self.num_genre)*g
                data.append(int(ct))
        random.shuffle(data)
        bs_data = []
        
        for b in range(len(data)//5):
            temp = [0 for _ in range(self.num_contents)] 
            for c in range(5):
                idx = data[5*b+c]
                temp[idx] += 1
            bs_data.append(temp)
            
        return bs_data
    
    def run(self):
        if self.ES.ESe != 0:
            data = self.set_data()
            self.train(self.model, data, num_epochs=10)
            
        elif self.ES.ESe == 0:
            self.model = self.load_global_model()
            
        self.train(self.model, self.data, num_epochs=10)
        self.ES.clients[self.ES.count%self.ES.size] = self.model.state_dict()
        self.ES.count += 1
