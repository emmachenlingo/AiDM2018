import random
import matplotlib.pyplot as plt 
from math import log
import pickle
import numpy as np

def generate_random(length):
    values = random.sample(range(0, 0xffffffff),k=length)
    return values

bindiv = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(32)] ) )
def run_of_zeros(h,b):
    binary_value=bindiv(h)[0:(32-b)]
    trailing_zeros=len(binary_value)-len(binary_value.rstrip('0'))
    ind_zero=31-b-trailing_zeros
    return(ind_zero)

def get_bitmap_index(h,b):
    bit_index=bindiv(h)[(32-b):]
    return(int(bit_index,2))

def least_sig_bit(bitmap,b):
    least_sig=np.zeros(bitmap.shape[0])
    for j in range(bitmap.shape[0]):
        least_sig[j]=31-b-max(np.where(bitmap[j]==0)[0])
    return(least_sig)

def estimate_PCSA (data,b):
    m = 2**b # with b in [4...16]
    bitmaps = [[0]*(32-b)]*m # initialize m 32bit wide bitmaps to 0s
    bitmaps_array=np.asarray(bitmaps)
    for i in range(0,len(data)):
        bitmap_index = get_bitmap_index( data[i],b ) # binary address of the rightmost b bits
        run_length = run_of_zeros( data[i],b ) # length of the run of zeroes starting at bit b+1
        bitmaps_array[bitmap_index,run_length ] = 1 # set the bitmap bit based on the run length observed
 
##############################################################################################
# Determine the cardinality
    phi = 0.77351
    DV = (m / phi) * (2**(sum(least_sig_bit(bitmaps_array,b)) / m))
    return(DV)

for k in range(5,11):
    RAE=[]
    values=[]
    Ns=[]
    N=0
  
    while N<10000000:
        if N<100000:
             step = 5000
        elif N<1000000:
             step = 50000
        else:
             step = 500000
        values += generate_random(step)
        N = len(set(values))
        Ns.append(N)
        est_N= estimate_PCSA(values,k)
        RAE.append(abs(est_N-N)/N*100)
    print(len(Ns))
    plt.xlim(4000, 1.1e7)
    plt.plot(Ns, RAE, label='k={0}'.format(k), lw=2)
