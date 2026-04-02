def tv_dist(pg, pc):
    ans = 0 
    for i in range(len(pg)):
        ans += abs(pg[i]-pc[i])
    ans /= 1/2
    
def inference(model, info, mode = 'split'):
    _, num_genre, num_contents, _, _ = info
    
    if mode == 'split':
        num_contents /= num_genre
        
    temp = [0 for i in range(int(num_contents))]
    for i in range(10000):
        t1 = model.sample().tolist()[0]
        for i in range(len(temp)):
            temp[i]+=t1[i]
    for i in range(len(temp)):
        temp[i]/=10000
    return temp
