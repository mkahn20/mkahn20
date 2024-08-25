import pickle
import matplotlib.pyplot as plt
from Train_data import *
from Fed_VAE import *
from VAE import *
from Inference import *

num_user     = 100
num_genre    = 10
num_contents = 2000
num_request  = 1000
num_zone     = 2
device       = "cuda:0"
info = [num_user, num_genre, num_contents, num_request, num_zone]

if __name__=="__main__":
    with open("./1000_0317_raw.pkl", "rb") as f:
        src = pickle.load(f)
        
    src = train_data(info, src, bs = 5, option =  "default")
    
    w1, w2, w3, w4 = fed_VAE(src, info, device)
    
    # with open("0722_w.pkl", "wb") as f:
    #     pickle.dump([w1,w2,w3,w4], f)
        

    ans = [] ## 추론 저장용
    for idx, w in enumerate(w3):
        m1 = VAE(info, 'global').to("cuda:0")
        m1.load_state_dict(w)
        t1 = inference(m1, info, mode = 'global')
        ans.append(t1)
        #plt.title("Client ID: "+str(idx))
        #plt.stem([i for i in range(len(t1))], t1)
        #plt.show()
        
    # with open("inference_0722.pkl", "wb") as f:
    #     pickle.dump(ans, f)