import pickle
from Utils import *
#from Data_gen import *
from Train_data import *
from VAE import *

num_user     = 100
num_genre    = 10
num_contents = 2000
num_request  = 1000
num_zone     = 2
info = [num_user, num_genre, num_contents, num_request, num_zone]
ut = Utils(info)

if __name__ == '__main__':
    with open("Non_shift_per-ver1", "rb") as f:
        p1w = pickle.load(f)
    _, _, wv1 = p1w
    with open("Non_shift_per-ver2", "rb") as f:
        p2w = pickle.load(f)
    _, _, wv2= p2w
    
    pv1 = [[] for _ in range(num_zone)]
    pv2 = [[] for _ in range(num_zone)]
    for z in range(num_zone):
        for idx, i in enumerate(wv1[z]):
            pv1[z].append(ut.sampler(i))
            print("Ver1 Zone:", z,"user:",i,"Done")
    for z in range(num_zone):    
        for idx, i in enumerate(wv2[z]):
            pv2[z].append(ut.sampler(i))
            print("Ver2 Zone:", z,"user:",i,"Done")
        
    with open("0321_inference.pkl", "Wb") as f:
        pickle.dump([pv1, pv2], f)
