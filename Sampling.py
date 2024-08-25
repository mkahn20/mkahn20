import numpy as np

def CDF(p):    
    dist = [0 for i in range(len(p))]
    for i in range(len(p)):
        for j in range(0, i+1):
            dist[i] += p[j]

    dist = np.array(dist)
    return dist

def binary_search(cdf, target, left, right):
    while left < right:
        mid = (left + right) // 2
        if cdf[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

def inverse_transform_sampling(cdf, n_samples, left, right, seed):
    np.random.seed(seed)
    u = np.random.rand(n_samples)
    samples = [binary_search(cdf, uu, left, right) for uu in u]
    return samples