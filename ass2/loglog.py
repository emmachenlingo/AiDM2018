import random
import matplotlib.pyplot as plt 
from math import log
import pickle

def trailing_zeroes(num):
  """Counts the number of trailing 0 bits in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

def estimate_cardinality(values, k):
  """Estimates the number of unique elements in the input set values.

  Arguments:
    values: An iterator of hashable elements to estimate the cardinality of.
    k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
  """
  num_buckets = 2 ** k
  max_zeroes = [0] * num_buckets
  for h in values:
    bucket = h & (num_buckets - 1) # Mask out the k least significant bits as bucket ID
    bucket_hash = h >> k
    max_zeroes[bucket] = max(max_zeroes[bucket], trailing_zeroes(bucket_hash))
  return 2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402, max_zeroes

#simulate the data stream by generating a long sequence of random 32-bit long integers
def generate_random(length):
  values = [random.randint(0, 0xffffffff) for i in xrange(length)]
  return values

fig = plt.figure(figsize=(10,6))
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
    est_N, registers = estimate_cardinality(values,k)
    RAE.append(abs(est_N-N)/N*100)
  print(len(Ns))
  plt.xlim(4000, 1.1e7)
  plt.plot(Ns, RAE, label='k={0}'.format(k), lw=2)

plt.xlabel('Cardinality (N)')
plt.ylabel('% RAE')
plt.legend()
plt.savefig('log-RAE.png')
plt.close(fig)

fig = plt.figure(figsize=(10,6))
plt.hist(registers, bins=range(5,25), facecolor='k', alpha=0.5)
plt.axvline(float(sum(registers))/2**k,color='r', label='True value')
plt.xlabel('Register value (R)')
plt.legend()
plt.savefig('log-hist.png')
plt.close(fig)
