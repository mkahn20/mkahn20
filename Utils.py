from FL import *
from VAE import *
from Train_data import *
from Data_gen import *
import torch
import matplotlib.pyplot as plt

class Utils():
    def __init__(self, info):
        self.info = info
        
    def sampler(self, weight):
        model = VAE().to("cuda:0")
        model.load_state_dict(weight)
        prob = self.inference(model)
        return prob
    
    def test(self, p1t1, p1t2, p2t1, p2t2, gt1, gt2, num_iter = 100):
        num_user, num_genre, num_contents, num_request, num_zone = self.info
        rp1t1 = torch.tensor(p1t1).topk(num_contents)[1].tolist()
        rp1t2 = torch.tensor(p1t2).topk(num_contents)[1].tolist()
        rp2t1 = torch.tensor(p2t1).topk(num_contents)[1].tolist()
        rp2t2 = torch.tensor(p2t2).topk(num_contents)[1].tolist()
        rgt1 = torch.tensor(gt1).topk(num_contents)[1].tolist()
        rgt2 = torch.tensor(gt2).topk(num_contents)[1].tolist()
        dg = Data_gen(num_user = num_user
                ,num_genre = num_genre,
                num_contents = num_contents,
                num_request  = num_request,
                id = [[i for i in range(int(num_user))], [i for i in range(int(num_user))]])

        p1, p2 = dg.run(seed = 0)
        r1, r2 = [], []

        for s in range(1, num_iter+1):
            ru1 = inverse_transform_sampling(cdf = CDF(p1[27]), 
                                        n_samples = int(num_request), 
                                        left = 0, right = num_contents - 1, seed = s)
            ru2 = inverse_transform_sampling(cdf = CDF(p2[37]), 
                                        n_samples = int(num_request), 
                                        left = 0, right = num_contents - 1, seed = s)
            r1.append(ru1)
            r2.append(ru2)
        print("Global VS Per-ver1 VS Per-ver2: Target User")
        for topk in [5, 10, 15, 20, 25, 30, 100]:
            rs = [[] for _ in range(6)]
            for s in range(num_iter):
                temp = [0 for _ in range(6)]
                temp[0], temp[1], temp[2] = self.chker(r1[s], rgt1, topk), self.chker(r1[s], rp1t1, topk), self.chker(r1[s], rp2t1, topk)
                temp[3], temp[4], temp[5] = self.chker(r2[s], rgt2, topk), self.chker(r2[s], rp1t2, topk), self.chker(r2[s], rp2t2, topk)
                for i in range(6):
                    rs[i].append(temp[i])

            print("Topk:",topk)
            za1 = np.array(rs[0]).mean()
            zb1 = np.array(rs[1]).mean()
            zc1 = np.array(rs[2]).mean()
            print("Zone1:",round(za1,2)," ",round(zb1,2)," ",round(zc1,2))

            za2 = np.array(rs[3]).mean()
            zb2 = np.array(rs[4]).mean()
            zc2 = np.array(rs[5]).mean()
            print("Zone2:", round(za2,2) ," ",round(zb2,2)," ",round(zc2,2))
            print('-'*50)
        
    def chker(self, src, trg, topk):
        s = 0
        t = trg[0:topk]
        for i in t:
            for j in src:
                if i == j:
                    s+=1
        return s
    def prep_chk(self, p, type = 'user'):
        num_user, num_genre, num_contents, num_request, num_zone = self.info
        g = [0 for _ in range(num_genre)]
        if type == 'user':
            for i in p:
                g[i//(num_contents//num_genre)]+=1
            
        elif type == 'global':
            for i in p:
                for j in i:
                    g[j//(num_contents//num_genre)]+=1
        g = torch.tensor(g).topk(num_genre)[1].tolist()
        return g

    def inference(self, model):
        num_user, num_genre, num_contents, num_request, num_zone = self.info
        temp = [0 for i in range(num_contents)]
        for i in range(100000):
            t1 = model.sample().tolist()[0]
            for i in range(len(temp)):
                temp[i]+=t1[i]
        for i in range(len(temp)):
            temp[i]/=100000
        # plt.stem([i+1 for i in range(len(temp))],temp,'r',label = 'prediction')
        # plt.title("Probability Distribution of Movie Popularity")
        # plt.xlabel("Movie ID")
        # plt.ylabel("Probability")
        # plt.legend(loc ='best')
        # plt.show()
        return temp
        
    def synt_model(self, model, gmodel, data, device = "cuda:0"):
        num_user, num_genre, num_contents, num_request, num_zone = self.info
        optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
        s = 0
        for i in data:
            s += i
            
        for i in range(len(data)):
            data[i] /= s
        data = torch.tensor(data, dtype=torch.float32).to(device)
        for epoch in range(100):
            for _ in range(10):
                model.train()
                train_loss = 0.0
                #batch = [batch]
                batch = data
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
                
            for _ in range(10):
                model.train()
                train_loss = 0.0
                #batch = [batch]
                batch = torch.tensor(gmodel.sample()).to(device)
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
                
        return model.state_dict()

    def fed_VAE(self, src, pu, shift, psn):
        num_user, num_genre, num_contents, num_request, num_zone = self.info
        # hyper parameter
        n_ES       = 2
        n_client   = num_user
        ES_epoch    = 100
        ESs = []
        clients = [[ None for i in range(n_client)] for j in range(n_ES) ]
        load1 = [num_request for i in range(num_user)]
        print('Initialize Dataset...')
        for i in range(n_ES):
            gs = [0  for i in range(num_genre)]
            ESs.append(ES(size=n_client, load = load1, device = "cuda:0", shift = shift, psn = psn))
            for j in range(n_client):
                # if(i == 0):
                #     src_u = src[0]
                #     cs = prep_chk(pu[i][j], 'user')
                # if(i == 1):
                #     src_u = src[1]
                #     cs = prep_chk(pu[i][j], 'user')

                clients[i][j]=Client(rank=j, 
                                    data = src[i][j], 
                                    local_epoch=10, 
                                    cr = 0.2,
                                    cs = self.prep_chk(pu[i][j], 'user'),
                                    gs = gs, 
                                    ES = ESs[i] )

        for ESe in range(ES_epoch):
            print('\n================== Edge Server Epoch {:>3} =================='.format(ESe + 1))

            for ESn in range(n_ES):
                #print("================= Edge Server :",ESn,"process =================")
                for c in clients[ESn]:
                    c.run()
                    
                ESs[ESn].aggregate()
        weight1 = ESs[0].global_weight()
        weight2 = ESs[1].global_weight()
        wresult = [[] for _ in range(num_zone)]
        if psn == "per-ver2":
            if shift:
                for u in range(num_user):
                    gs1 = self.prep_chk(pu[0], 'global')
                    gs2 = self.prep_chk(pu[1], 'global')
                    nm1 = VAE().to("cuda:0")
                    nm2 = VAE().to("cuda:0")
                    gm1 = VAE().to("cuda:0")
                    gm2 = VAE().to("cuda:0")
                    gm1.load_state_dict(weight1)
                    gm2.load_state_dict(weight2)
                    dn1 = [0 for _ in range(num_contents)]
                    dn2 = [0 for _ in range(num_contents)]
                    for cidx in pu[0][u]:
                        gid  = cidx//(num_contents//num_genre)
                        cid  = cidx%(num_contents//num_genre)
                        ngid  = gs1.index(gid)
                        nid  = ngid*(num_contents//num_genre) + cid
                        dn1[nid]+=1
                        
                    for cidx in pu[1][u]:
                        gid  = cidx//(num_contents//num_genre)
                        cid  = cidx%(num_contents//num_genre)
                        ngid  = gs2.index(gid)
                        nid  = ngid*(num_contents//num_genre) + cid
                        dn2[nid]+=1
                            
                    # nww1 = synt_model(model = nm1, gmodel = gm1, data = src[0][0])
                    # nww2 = synt_model(model = nm2, gmodel = gm2, data = src[1][-1])
                    # weight3 = nww1
                    # weight4 = nww2
                    # wresult[0].append(weight3)
                    # wresult[1].append(weight4)
                
            else:
            # cs1 = prep_chk(pu[0][0], 'user')
            # cs2 = prep_chk(pu[1][-1], 'user')
                for u in range(num_user):
                    nm1 = VAE().to("cuda:0")
                    nm2 = VAE().to("cuda:0")
                    gm1 = VAE().to("cuda:0")
                    gm2 = VAE().to("cuda:0")
                    gm1.load_state_dict(weight1)
                    gm2.load_state_dict(weight2)
                    dn1 = [0 for _ in range(num_contents)]
                    dn2 = [0 for _ in range(num_contents)]
                    for i in src[0][u]:
                        for idx, j in enumerate(i):
                            dn1[idx]+=j
                
                    for i in src[1][u]:
                        for idx, j in enumerate(i):
                            dn2[idx]+=j

                    nww1 = self.synt_model(model = nm1, gmodel = gm1, data = dn1)
                    nww2 = self.synt_model(model = nm2, gmodel = gm2, data = dn2)
                    weight3 = nww1
                    weight4 = nww2
                    wresult[0].append(weight3)
                    wresult[1].append(weight4)
                    
        elif psn == "per-ver2":
            for z in range(num_zone):
                for u in range(num_user):
                    wresult[z].append(ESs[z].clients[u])
                

        return weight1, weight2, wresult


