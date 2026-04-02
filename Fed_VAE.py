from Client import *
from ES import *

def fed_VAE(src, info, device = "cuda:0"):
    num_user, _, _, num_request, num_zone = info
    n_ES       = num_zone
    n_client   = num_user
    ES_epoch    = 100
    ESs = []
    clients = [[ None for i in range(n_client)] for j in range(n_ES) ]
    load1 = [num_request for i in range(num_user)]
    print('Initialize Dataset...')
    for i in range(n_ES):
        ESs.append(ES(size=n_client, load = load1, device = device, info = info))
        for j in range(n_client):
            clients[i][j] = Client(rank=j, 
                                 data = src[i][j], 
                                 local_epoch=10, 
                                 ES = ESs[i] )

    for ESe in range(ES_epoch):
        print('\n================== Edge Server Epoch {:>3} =================='.format(ESe + 1))

        for ESn in range(n_ES):
            #print("================= Edge Server :",ESn,"process =================")
            for c in clients[ESn]:
                c.run()
        
                
            ESs[ESn].aggregate()
            ESs[ESn].split_model()
            ESs[ESn].inference_dist()
            ESs[ESn].ESe += 1 ## seed increasing
            
    weight1 = ESs[0].global_weight()
    weight2 = ESs[1].global_weight()
    weight3 = ESs[0].clients
    weight4 = ESs[1].clients
    

    return weight1, weight2, weight3, weight4