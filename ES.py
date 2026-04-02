import copy
from torch import optim
from VAE import *
from Inference import *
from Train_data import *

class ES():
    def __init__(self, size, load, device, info):
        self.size = size
        self.model  = VAE(info, 'global').to(device)
        self.gmodels = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3,  amsgrad=True)
        self.clients = [None]*size
        self.count   = 0
        self.ESe     = 0 
        self.load    = load
        self.load_s  = 0
        self.device = device
        self.dist = None
        self.info = info
        self.num_user, self.num_genre, self.num_contents, self.num_request, self.num_zone = info
        for i in load:
            self.load_s+=i
    
    def inference_dist(self):
        temp = []
        for i in self.gmodels:
            temp.append(inference(i, self.info, mode ='split'))
        self.dist = temp
    
    def split_model(self):
        models = [VAE(self.info, 'split').to("cuda:0") for _ in range(self.num_genre)]
        gdict = []
        w = self.model.state_dict()
        ptr = self.num_contents//self.num_genre
        for g in range(self.num_genre):
            wt = copy.deepcopy(w)
            w1 = w["fc1.weight"][:,g*ptr:(g+1)*ptr]
            w2 = w["fc4.weight"][g*ptr:(g+1)*ptr,:]
            w2b = w["fc4.bias"][g*ptr:(g+1)*ptr]
            wt["fc1.weight"] = w1
            wt["fc4.weight"] = w2
            wt["fc4.bias"] = w2b
            gdict.append(wt)
            
        for g in range(self.num_genre):
            models[g].load_state_dict(gdict[g])
        
        self.gmodels = models
        
        
    def average_weights(self,clients):
        ws = copy.deepcopy(clients[0])
        for idx, info in enumerate(clients[1:]):
            for key in info:
                ### 수정했음 ### 0722_Client 0 -> global로 바뀌는 문제 해결
                ## 정상 작동 안되면 연락주세요.
                if idx == 0:
                    ws[key] = self.load[idx+1]*info[key] + self.load[idx]*ws[key]
                else:
                    ws[key] += self.load[idx+1]*info[key]
                    
        for key in ws:
            ws[key]=ws[key]/self.load_s
        weights=ws
        return weights

    def aggregate(self):
        weights_info = self.clients
        weights = self.average_weights(weights_info)
        self.model.load_state_dict(weights)
        
    def global_weight(self):
        weights = self.model.state_dict()
        return weights
    def sample(self):
        return self.model.sample()