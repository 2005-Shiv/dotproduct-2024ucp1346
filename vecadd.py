import numpy as np
from multiprocessing import Pool
#small modification
def parallel_dot_product(args):
    a, b, start, end = args
    return np.sum(a[start:end] * b[start:end])

def dot_product_parallel(a, b, num_processes=4):
    n = len(a)
    chunk_size = n // num_processes
    args = [(a, b, i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    
    with Pool(num_processes) as pool:
        result = sum(pool.map(parallel_dot_product, args))
    return result

