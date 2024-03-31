import torch
import copy
from torch import optim
from torch import nn
from torch.autograd import Variable
from VAE import *

num_user     = 100
num_genre    = 10
num_contents = 600
num_request  = 300
num_zone     = 2
info = [num_user, num_genre, num_contents, num_request, num_zone]

class ES():
    def __init__(self, size, load, device, shift, psn):
        self.size = size
        self.model = VAE().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3,  amsgrad=True)
        self.clients = [None]*size
        self.count=0
        self.load =load
        self.load_s = 0
        self.device = device
        self.shift  = shift
        self.psn = psn
        for i in load:
            self.load_s+=i
            

    def average_weights(self,clients):
        for idx, info in enumerate(clients[1:]):
            for key in info:
                #clients[0][key]=info[key] + clients[0][key]
                if(idx == 0):
                    clients[0][key]=self.load[idx+1]*info[key] + self.load[idx]*clients[0][key]
                else:
                    clients[0][key]=self.load[idx+1]*info[key] + clients[0][key]
                    
        for key in clients[0]:
            #clients[0][key]=clients[0][key]/self.size  
            clients[0][key]=clients[0][key]/self.load_s
        weights=clients[0]
        return weights


## Optinal 
    def test(self):
        test_correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        return test_correct / len(self.test_loader.dataset)

    def aggregate(self):
        ## overlapped client weight gamma needed 
        weights_info = self.clients
        weights = self.average_weights(weights_info)
        self.model.load_state_dict(weights)
    def global_weight(self):
        weights = self.model.state_dict()
        return weights
    def sample(self):
        return self.model.sample()

class Client(object):
    def __init__(self, rank, data, local_epoch, cr, cs, gs, ES):
        # seed
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        self.ground = 0
        self.model = VAE().to(ES.device)
        self.rank = rank
        self.local_epoch = local_epoch
        self.ES = ES
        self.ws = None
        self.data = data
        self.cr   = cr
        self.cs   = cs
        self.gs   = gs
    
    def load_global_model(self):
        model = VAE().to(self.ES.device)
        model.load_state_dict(self.ES.model.state_dict()) 
        return model
    
    def aggregate(self, model):
        # wc = self.model.state_dict()
        #wc = self.param_mapper(self.model, self.cs, self.gs, forward = True)
        wc = self.model.state_dict()
        wg = model.state_dict()
        ws1 = copy.deepcopy([wc, wg])
        for idx, info in enumerate(ws1[1:]):
            for key in info:
                if(idx == 0):
                    ws1[0][key]=self.cr*info[key] + (1-self.cr)*ws1[0][key]
                else:
                    ws1[0][key]= info[key] + ws1[0][key]
        
        return ws1[0]
                    

    def train(self, model):
        # update local model
        optimizer = optim.Adam(model.parameters(), lr=1e-3,  amsgrad=True)
        num_epochs = 10
        batch = self.data
        for _ in range(num_epochs):
            model.train()
            train_loss = 0.0
            #batch = [batch]
            batch = torch.tensor(batch, dtype=torch.float32).to(self.ES.device)
            #inputs = data.to('cuda:0')
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
        #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss ))
        
        weights = model.state_dict()
        
        return weights
    
    def param_mapper(self, model, cs, gs, layer = 'fc4', forward = True):
        lw = layer + '.weight'
        lb = layer + '.bias'
        weight = copy.deepcopy(model.state_dict())
    
        if forward:
            for id, i in enumerate(cs):
                idx = gs.index(i)
                for j in range(num_contents//num_genre):
                    weight[lb][idx*(num_contents//num_genre)+j] = model.state_dict()[lb][id*(num_contents//num_genre)+j]
                    weight[lw][idx*(num_contents//num_genre)+j] = model.state_dict()[lw][id*(num_contents//num_genre)+j]

        else:
            for id, i in enumerate(gs):
                idx = cs.index(i)
                for j in range(num_contents//num_genre):
                    weight[lb][idx*(num_contents//num_genre)+j] = model.state_dict()[lb][id*(num_contents//num_genre)+j]
                    weight[lw][idx*(num_contents//num_genre)+j] = model.state_dict()[lw][id*(num_contents//num_genre)+j]
                    
                
        return weight
        
    def run(self):
        if self.ES.psn == "per-ver1":
            model = self.load_global_model()
            
            if self.ES.shift:
                model.load_state_dict(self.param_mapper(model, self.cs, self.gs, forward = False))
                if self.ground == 0:
                    weights = self.train(model)
                    self.model.load_state_dict(weights)
                    self.ground += 1
                    
                else:
                    weights = self.aggregate(model)
                    model.load_state_dict(weights)
                    #self.ws = copy.deepcopy(weights)
                    weights = self.train(model)            
                    self.model.load_state_dict(weights)
                
                weights = self.param_mapper(model, self.cs, self.gs, forward = True)
            
            else:
                if self.ground == 0:
                    weights = self.train(model)
                    self.model.load_state_dict(weights)
                    self.ground += 1
                    
                else:
                    weights = self.aggregate(model)
                    model.load_state_dict(weights)
                    #self.ws = copy.deepcopy(weights)
                    weights = self.train(model)            
                    self.model.load_state_dict(weights)
                
        elif self.ES.psn == "global" or "per-ver2":
            model = self.load_global_model()
            if self.ES.shift:
                model.load_state_dict(self.param_mapper(model, self.cs, self.gs, forward = False))
                weights = self.train(model)
                weights = self.param_mapper(model, self.cs, self.gs, forward = True)
                
            else:
                weights = self.train(model)
            
        self.ES.clients[self.ES.count%self.ES.size] = weights
        self.ES.count += 1
