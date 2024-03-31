import torch
import random

def train_data(info, src_rw, bs = 5, option = "default"):
    num_user, num_genre, num_contents, num_request, num_zone = info
    rk_rw = [[[] for _ in range(num_user)] for _ in range(num_zone)]
    
    ## Default -> Do not reflect user preference : using original index
    ## So, the first conditional state means that shuffling the original index reflects user preference.
    if option != "default":
        for i in range(num_zone):
            for j in range(num_user):
                temp_rk2 = torch.tensor([0 for _ in range(num_genre)])
                for k in range(num_request):
                    temp_rk2[src_rw[i][j][k]//(num_contents//num_genre)]+=1
                rk_rw[i][j]+=(temp_rk2.topk(num_genre)[1].tolist())
                
                
        src_rw2 = [[[0 for _ in range(num_contents)] for _ in range(num_user)] for _ in range(num_zone)]
        for i in range(num_zone):
            for j in range(num_user):
                for k in (src_rw[i][j]):
                    gid = k//(num_contents//num_genre)
                    cid = k%(num_contents//num_genre)
                    n_gid = rk_rw[i][j].index(gid)
                    idx = n_gid*(num_contents//num_genre)+cid
                    src_rw2[i][j][idx]+=1
    
        src_q = [[[] for _ in range(num_user)] for _ in range(num_zone)]
        for i in range(num_zone):
            for j in range(num_user):
                for k in range(num_contents):
                    if src_rw2[i][j][k]!=0:
                        for s in range(src_rw2[i][j][k]):
                            src_q[i][j].append(k)
                            
    else:
        src_q = src_rw
    
    for i in range(num_zone):
        for j in range(num_user):
            random.shuffle(src_q[i][j])
    
    ## Make Mini-Batch
    src = src_q
    src_m = [[[] for _ in range(num_user)] for _ in range(num_zone)]
    for z in range(num_zone):
        for u in range(num_user):
            for b in range(num_request//bs):
                temp = [0 for _ in range(num_contents)] 
                for c in range(bs):
                    idx = src[z][u][bs*b+c]
                    temp[idx] += 1
                src_m[z][u].append(temp)
    
    return src_m