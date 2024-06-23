
# SHAllenge

Code for https://shallenge.quirino.net/ https://news.ycombinator.com/item?id=40683564

To build, run `make`. To do so in a docker container, run:
```bash
docker run --rm -w /workspace -v .:/workspace --user $(id -u):$(id -g) nvidia/cuda:12.4.0-devel-ubuntu22.04 make
```

To run:
```bash
$ ./build/shallenge --seed=1337 --hashes=0.1 --block_size=1024 --grid_size=48
Number of devices: 1
Device 0: NVIDIA GeForce RTX 3070 Ti
  Compute capability: 8.6
  Total global memory: 8589410304 bytes
  Shared memory per block: 49152 bytes
  Registers per block: 65536
  Warp size: 32
  Max threads per block: 1024
  Max threads per multiprocessor: 1536
  Number of multiprocessors: 48
Register usage: 56
Shared memory per block: 0 bytes
Occupancy: 24576 threads per multiprocessor
Seed: 1337
Hashes in total: 0.100 TH
Grid size 48, Block size: 1024, Threads: 49152
Kernel launches: 2
Hash rate: 5.05 GH / s
Best nonce: melifkabmnmnjjda
Best hash: 00000000 01a5ba71 c1fb13a2 02d8ad30 0784d9b7 ded85f69 02f5fd4f c70c53c6
Iteration 1 of 2 completed.
Elapsed time: 10.20 s
Hash rate: 5.00 GH / s
Iteration 2 of 2 completed.
Elapsed time: 20.52 s
....

```
A lot of stuff is hardcoded right now, including my username.
Can also do it in docker: 
```bash
docker run --gpus all --rm -w /workspace -v ./build:/workspace --user $(id -u):$(id -g) nvidia/cuda:12.4.0-runtime-ubuntu22.04 /workspace/shallenge --seed 10 --hashes 0.1
```

The code can be run remotely with multiple GPUs using modal.com:
```
python3 -m venv ./env
./env/bin/pip install modal
./env/bin/modal run --detach model_remote.py
```


To profile, compile the program with --lineinfo by running 
```bash
make PROFILE=1 ncu
```

A table of hash rates. Grid size set to 2x the number of SMs for each arch.

| GPU      | Hash Rate (GH/s) |
| -------- | ---------------- |
| RTX 3070 | 4.9              |
| RTX 3090 | 6.7              |
| RTX 4090 | 18.4             |
| H100     | 14.3             |

Calculations on number of hashes needed to have a 50% probability of getting a hash with N or more zeros in its prefix.

$$
p = (1 - \frac{1}{2 ^ N})^ H
$$

Consider 
$$ \ln (1 - x)\approx  -x $$
when $x$ is small, we get

$$
H = - 2 ^ N \ln p
$$

The top entry on the leader board has 56 leading zeros in its hash. So we need to run $4.9 * 10^{16}$ hashes to have a 50% probability of getting a hash with as many or more zeros. For my setup with 8 + 9 = 17 RTX 4090s, each with a hash rate of 14 GH /s, or a total hash rate of 240 GH / s, it'll take 57 hours to have a 50% chance of getting number one place on the leader board. The 4090s cost me $3.3 per hour in total, running 57 hour costs $188.









