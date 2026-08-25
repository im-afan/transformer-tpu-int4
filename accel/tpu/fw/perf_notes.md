# Kernel performance notes

- current kernel performance (double-buffered, dma overlapping with compute when possible)
```
core active time over 32 generation(s) @ 12 MHz:
  per problem   mean 2220.898 ms   min 2220.893 ms   max 2220.902 ms
  per token     mean   69.403 ms (832837 core clocks over 32 token(s))
  unit busy     mxu 45.5%  mload 16.0%  vpu 2.8%  dma 54.1%  swait 0.0%  vmm 0.0%  idlec 17.5%  qfull 0.0%  ovlap 19.9%

prefill vs decode, by subtraction (GEN=1 against GEN=32):
                               clocks        ms       mxu   mload     vpu     dma   idlec   ovlap
  prefill (32 rows)           1768763   147.397     48.6%    7.5%   10.2%   30.9%   16.5%    6.2%
  one decode step (M=1)        802645    66.887     45.2%   16.6%    2.3%   55.8%   17.5%   20.9%
  whole generation           26650766  2220.897     45.5%   16.0%    2.8%   54.1%   17.5%   19.9%

  a prompt token costs 55274 clocks in the prefill, 802645 in a decode step (14.5x)

accuracy over 32 problems, generating 32 token(s):
                              exact-sequence     token
  FPGA, generating                     0.00%     7.42%
  QAT model, greedy                    0.00%     6.35%

device and model generated the same sequence on 0 of 32 problems
```
- model: L=4, d=128, f=512. total transformer params = 12ld^2 = 394 KB int4
- prefill: 
    - device: 
        - memory: 1 byte / clk
        - mxu: 8*8 = 64 INT4 OP / clk
    - memory intensity: params = 12ld^2 = 394 KB 
        - => 394,000 clocks to load all weights
        - kv cache: 2 * T * L * D = 32 KB
        - => 32,000 clocks to write KV, total 426,000 clocks
    - arithmetic intensity: ops = 8lbd^2 (ffn) + 4lbd^2 (mha) = 25,000,000 INT4 ops
        - => 390,625 clocks to finish compute
    - even with no overlap, theoretically only ~800k clocks for prefill. 
