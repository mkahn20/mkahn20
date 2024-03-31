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
# dg = Data_gen(num_user = num_user
#               ,num_genre = num_genre,
#               num_contents = num_contents,
#               num_request  = num_request,
#               id = [[i for i in range(int(num_user))], [i for i in range(int(num_user))]])

if __name__ == '__main__':
    with open("1000_0317_raw.pkl", "rb") as f:
        pu = pickle.load(f)
    src = train_data(info, pu, bs = 5, option =  "default")
    sets = [[False, "per-ver1"], [False, "per-ver2"], [False, "global"]]
    for s in sets:
        print(s[0])
        w1, w2, w3, w4 = ut.fed_VAE(src = src, pu = pu, shift = s[0], psn = s[1])
        w = [w1, w2, w3, w4]
        if s[0]:
            title = "Shift_" + s[1]
        else:
            title = "Non_shift_"+ s[1]
            
        with open(title, "wb") as f:
            pickle.dump(w, f)
