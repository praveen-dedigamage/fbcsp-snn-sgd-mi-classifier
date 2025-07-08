import torch
import time

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n.sqrt()) + 1):
        if n % i == 0:
            return False
    return True

# Vectorized method using broadcasting (GPU-compatible)
def find_primes_torch(limit, device):
    numbers = torch.arange(2, limit + 1, device=device)
    is_prime = torch.ones_like(numbers, dtype=torch.bool)
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i - 2]:  # offset for 2-based indexing
            is_prime[(numbers % i == 0) & (numbers != i)] = False
    return numbers[is_prime]

limit = 100_000_000

# Run on CPU
start = time.time()
primes_cpu = find_primes_torch(limit, device='cpu')
cpu_time = time.time() - start
print(f"✅ CPU found {len(primes_cpu)} primes in {cpu_time:.4f} seconds")

# Run on GPU
if torch.cuda.is_available():
    start = time.time()
    primes_gpu = find_primes_torch(limit, device='cuda')
    gpu_time = time.time() - start
    print(f"🚀 GPU found {len(primes_gpu)} primes in {gpu_time:.4f} seconds")
else:
    print("❌ GPU not available.")
