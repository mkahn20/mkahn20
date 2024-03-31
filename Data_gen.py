import pickle
import torch.nn.functional as F
import random
from Sampling import *

class Data_gen():
    def __init__(self, num_user, num_genre, num_contents, num_request, id, ur = 1, lr = 1):
        # self.src = src
        self.ur   = ur
        self.lr   = lr
        self.id   = id
        self.num_user     = num_user
        self.num_contents = num_contents
        self.num_genre    = num_genre
        self.num_request  = num_request
        
        with open("./mt_zone1.pkl", "rb") as f:
            self.z1 = pickle.load(f)
    
        with open("./mt_zone2.pkl", "rb") as f:
            self.z2 = pickle.load(f)
        
    def set_ratio(self, zone):
        nu  = int(self.num_user*self.ur)
        u1 = random.sample([i for i in range(10000)], nu)
        
        #u2 = random.sample([i for i in range(10000)], nu)
        rk1 = []
        #rk2 = []
        for u in u1:
            rk1.append(zone[u])
        #for u in u2:
        #    rk2.append(z1[u])
        
        return rk1
    
    def contents_dist(self, q_m, gamma_m):
        n = self.num_contents//self.num_genre
        p_m = [0 for _ in range(n)]
        #q_m = 64
        #gamma_m = 5
        sum = 0
        for i in range(n):
            p_m[i] = (i+q_m)**(-gamma_m)
            sum += p_m[i]
        for i in range(n):
            p_m[i] /= sum

        return p_m

    ## Mzipf of Individual Genre prob
    def genre_dist(self, q_m, gamma_m):
        n = self.num_genre
        p_m = [0 for i in range(n)]
        sum = 0
        for i in range(n):
            p_m[i] = (i+q_m)**(-gamma_m)
            sum+=p_m[i]

        for i in range(n):
            p_m[i] /= sum
            
        return p_m
    
    def user_preference(self, id, rk):
        num_user = len(id)
        p_m = []
        for i in id:
            q_m   = 64 + i/2
            gamma = 5 + i/2
            p_m.append(self.genre_dist(q_m, gamma))
            p_c = []
            
        for i in range(self.num_genre):
            q_m   = 50 + 0.01*i/2
            gamma = 3 + 0.01*i/2
            p_c.append(self.contents_dist(q_m, gamma))

        p_g = [[0 for _ in range(self.num_genre)] for _ in range(num_user)]
        for u in range(num_user):
            for idx, g in enumerate(rk[u]):
                p_g[u][g-1] = p_m[u][idx]
            
        p_u = [[None for _ in range(self.num_contents)]for _ in range(num_user)]
        for u in range(self.num_user):
            for g in range(self.num_genre):
                for m in range(self.num_contents//self.num_genre):
                    p_u[u][(self.num_contents//self.num_genre)*g+m] = p_g[u][g]*p_c[g][m]

        return p_u
    def sampling(self, pu, id):
        z1_req = []
        num_user = len(id)
        for u in range(num_user):
            z1_req.append(inverse_transform_sampling(cdf = CDF(pu[u]), 
                                n_samples = int(self.lr*self.num_request), 
                                left = 0, right = 599))
        
        return z1_req
    def run(self):
        ## zone1
        rk1  = self.set_ratio(self.z1)
        p1   = self.user_preference(self.id[0], rk1)
        req1 = self.sampling(p1, self.id[0])
        
        ## zone2
        rk2  = self.set_ratio(self.z2)
        p2   = self.user_preference(self.id[1], rk2)
        req2 = self.sampling(p2, self.id[1])
        
        return req1, req2