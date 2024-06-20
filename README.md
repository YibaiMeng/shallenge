
# SHAllenge

Code for https://shallenge.quirino.net/ https://news.ycombinator.com/item?id=40683564

To build:

```
nvcc -O3 -o shallenge shallenge.cu 
```

To run:
```
$ ./shallenge --seed 0 --iter 100
Starting seed: 0
Iterations: 100
Threads per launch: 49152
Hashes in total: 5.15 TH
Next seed is 5153960755200
Hash rate: 4.7 GH / s
Best nonce: bcigbepchaaaaaaa
Best hash: 00000000 012b4fcf 7922ddb3 947d59a8 7673e1e1 fc1de9ce aba37511 dd81723d
Iteration 1 of 100 completed.
Elapsed time: 11.2 s
Hash rate: 4.7 GH / s
Iteration 2 of 100 completed.
Elapsed time: 22.1 s
....

```
A lot of stuff is hardcoded right now, including my username.

To profile, compile the program with --lineinfo:

```
nvcc -O3 -o shallenge shallenge.cu  --lineinfo
```

```
ncu -o profile -f --import-source true --set full --target-processes=application-only -k regex:find_lowest_sha256  ./shallenge --seed 0 --iter 1
```
