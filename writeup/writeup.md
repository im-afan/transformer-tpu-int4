# Transformers, TPUs, Kernels, and Lessons About AI-Assisted Development 

## Introduction

This was a summer project I made to learn the basics of ML systems. The overall goal of this project is to run a small transformer on my Cmod A7 FPGA board. Then, I want to derive some of the common results you'll find in ML (limited to a single device for now), and to get a feel for some of math behind model scaling and kernel optimization. I'm assuming you have basic knowledge on how transformer inference works (prefill, decode), but not the hardware side of things.

## Background

### What determines how fast our model is?

Before we get into hardware, we need to ask a seemingly obvious question: what makes a model/algorithm faster? In ML, this problem can be analyzed in 2 domains: compute and communications. Compute is how many raw operations we can do in some amount of time. For example, FLOP/s (floating point operations / sec) is a common metric for the compute capabilities of GPUs or other ML accelerators. So the total compute time of an algorithm can be calculated using $T_{\text{compute}} = \frac{\text{# FLOPs}}{\text{accelerator FLOPs/s}}$. 

On the other hand, we have communication. In single-device inference, this usually refers to the comms between the memory (HBM, DDR) and the accelerator's cache. In distributed inference, the comms between devices must also be considered. Similarly, if we know the memory bandwidth of our HBM or DDR, we can calculate our comms time as $T_{\text{comms}} = \frac{\text{Communication Bytes}}{\text{Memory Bytes/sec}}$

In most hardware, we assume that comms and compute run at the same time, so optimally, they are completely overlapped. So, our lower bound on the runtime of an algorithm is $\max(T_{\text{comms}}, T_{\text{compute}})$. 

### Matmuls

Matmuls have a property regarding compute and comms that makes them so special for ML. First, let us consider an $N\times N\times N$ matmul, in int8: $A[N, N] \cdot B[N, N] = C[N, N]$. In the best case, we have to load $2N^2$ bytes to our compute, then write back $N$ bytes to memory, which is a total of $3N^2$. For compute, we have to perform $N^3$ int8 multiplications. So the total arithmetic intensity of the algorithm is on the order of N, meaning it is more compute-demanding the larger our matmul gets. This is partly why matmuls are so essential to ML workloads: it is very easy to scale up our models by just throwing more compute at it to achieve larger matmuls.  


## Model Architecture / Goal

The overall goal is to run a transformer model on our design, while not making it completely fixed to a single architecture. We want to be able to write code to change the model architecture/dimensions, tweak our matmuls to be more efficient, or run unit tests to benchmark single layers or matmuls. This means that having fixed control flow set in hardware is not acceptable.

We will train and inference our model on predicting the next token in an addition sequence with a maxmium of 31 digits per input (64 token prefill and 64 token decode overall), which is a pretty simple but nontrivial task. 

For benchmarking, we use a standard Transformer architecture, with embedding dim $d=128$, ffn dim $d_{ff}=512$, 4 QKV heads, and 4 layers. However, we greatly simplify some operations to allow for ease of implementation. First, we replace gelu with relu activation. Softmax in attention is also replaced with a simple ReLU, which has a big effect on training but is fine for our task. LayerNorm is replaced with a tanh normalization. Finally, we train with no bias in the ffn. While these are pretty major simplifications, the overall architecture stays the same, and the original goal of analyzing transformer inference performance can still be achieved.

## Hardware

### Overview

In a transformer, we have 2 types of operations: matmuls, and elementwise ops like tensor addition and ReLU. The purpose of an accelerator is to load tensors from memory, do those operations, and write back the results. To make the most of our FPGA resources and memory, we choose to quantize activations and weights to int4. 

The memory heirarchy of our TPU is simple. We are using a Cmod A7 board, which includes an Artix-7 FPGA chip along with an external asynchronous SRAM chip (8 bit read, 8 ns access time). We use the SRAM chip to model our accelerator's external memory (HBM/DDR in a real accelerator), and the Artix-7's BRAM to act as an on-chip cache (scratchpad memory). 

There are 3 main units: MXU, VPU, and DMA. The MXU (matrix mutliply unit) handles the matrix multiplications, VPU (vector processing unit) handles vector operations such as activations and addition. Both units read from the scratchpad memory and write their results back to scratchpad. The DMA (direct memory access) handles transfers between scratchpad and external memory. Each unit is controlled by a softcore PicoRV32 processor, which we can write C firmware to dispatch instructions to each unit through AXI-based MMIO.

[TPU ARCHITECTURE DIAGRAM]

### Memory 

Our scratchpad memory is synthesized as simple dual-port BRAM, with 2 independent ports: read and write. Since it can be synthesized to basically an arbitrary bus width for our usage (1 x 32K bits to 72 x 512 bits), we don't need the blocks to be traditional banks by `address % bus width`. Instead, we can use a single block to represent a contiguous region of our scratchpad memory. This basically allows us to have as many ports accessing scratchpad as we want, with the limitation that 2 ports don't access the same region; this will be very useful in the design later.

[BRAM DIAGRAM]

Our DMA is very simple. Running at the Cmod A7's standard 12 MHz clock, the 8ns external RAM access time is basically instant; it arrives at the next clock. The DMA unit takes in a base address for scratchpad and external RAM, a row stride, and 2d matrix dimensions, and simply drives the ports of the external RAM and scratchpad to transfer data between them. However, there is one wrinkle: since our external SRAM is asynchronous and has an 8 ns access time, the WE (write-enable) signal needs to come only after the address and data signals have arrived and stabilized. This means that for each byte written, we need to turn WE on and back off again, at the negative edge of our clock. So our final bandwidth is 1 byte / clock for external -> scratchpad, and 0.5 byte / clock for scratchpad -> external.


### MXU & VPU

For matmuls, we use an output-stationary 8x8 systolic array; each PE keeps its partial sum, while moving its input values to the next PE to its right and below it. This allows us to perform an arbitrary 8xNx8 ($A[8,N] \cdot B[N,8])$) mamtul, as long as A and B fit in scratchpad. By default, all tensors are stored row-major in both external memory and scratchpad. Since our design has no fast way to transpose a matrix, we instead use a trick in the MXU to perform transposed matmuls such as $QK^T$ in attention. By default, A and B are fed into the systolic array by... [FINISH EXPLANATION, PROBABLY NEED A DIAGRAM FOR THIS TO BE EXPLAINED WELL] 

As a result, every clock, the systolic array reads $4N$ bits for matrix A and $4N$ bits for matrix B. This is where the scratchpad architecture comes in handy: as long as A and B are in different memory regions, they can be read at the same time, without interfering with ongoing DMA operations.

The VPU is pretty simple. Its inputs are the base address of a vector and the length of the operation. It repeatedly loads chunks of the vector from scratchpad, operates on them (add, relu, etc), and writes them back, until the operation is completed.

### Design Choices & Notes

We use PicoRV32 so that we can easily write firmware for different architectures. Not only does this allow for unit testing beyond just inference & different architectures, it allows us to easily experiment with optimizations later on without having to change the dataflow in hardware.

CPU issue overhead was also a concern when designing the architecture. When the CPU dispatches instructions, they enter a command queue for each unit, which are then executed asynchronous of the CPU execution order. For synchronization, the CPU can also poll each instruction queue's state. Furthermore, to minimize the effect of execution latency, the TPU operations are intentionally complex, allowing for instructions to span across large address ranges without needing more CPU executes.

## Firmware & Kernels

We now need to program the PicoRV32 core to perform inference. First, we define our programming scheme: 
- Since our external SRAM (512 KB) is much larger than our scratchpad (64 KB), we want to only use the scratchpad to handle the individual operands of a primtive. Every operation should read and write back to external memory.
- We will define many primitives such as matmul, tensor addition, ReLU, and then compose them together using RISCV's loop & branching capabilities to implement inference.

We start by implementing the basic instruction dispatch to the TPU. After that, we implement primitives, namely matmul and elementwise functions. 

### Optimizing A Matmul

We are lucky enough to have a relatively big scratchpad (64 KB) compared to our external SRAM (512 KB), which is a 1:8 ratio. With d=128, d_ff=512, and T=32 during prefill, our largest matmul, in the FFN, requires about $(128 * 512 + 2 * 32 * 128) / 2 = 36,864$ bytes (36 KB), which can fit entirely in our scratchpad! This means that we can basically always achieve the theoretical $NM + MK + NK$ byte loads for a matmul, since we don't need to load the same chunk of a matrix twice when tiling. 

We implement matmul as a tiled matmul over $8\times 8$ tiles in the output matrix. Here's the pseudocode for this kernel: 

```
inputs (in external memory): A[M, K], B[K, N]
output (in external memory): C[M, N]
allocate A_tile[8*S, K], B_tile[K, 8], C_tile[8, 8] in scratchpad, in different regions.
(Fit S such that it is maximized without overflowing the scratchpad.)

for i from 0 to M, with step 8*S:
    copy A[i:i+8*S][:] from external memory to A_tile
    for j from 0 to N, with step 8:
        copy B[:][j:j+8] from external memory to B_tile
        wait for all DMA to finish
        for k from 0 to 8*S, with step 8:
            tpu_matmul A_tile * B_tile -> C_tile
            wait for mxu to finish
            copy C_tile from scratchpad to C[i:i+8][j:j+8]
```

And if we instead want $C = AB^T$, we instead load `B[j:j+8][:]` to B_tile, and dispatch a transpose flag when calling the mxu.

This algorithm uses less scratchpad memory than the upper bound we just calculated, and it also supports matmuls that don't fully fit in the scratchpad by autofitting the S variable. However, that would cause inefficiency due to having to load each element of B more than once.

But this still isn't optimal! Remember that, optimally, our runtime is the max of the MXU and DMA time. We need a way to overlap our DMA accesses with the MXU decently. To do this, we instead allocate 2 regions for `B_tile` in scratchpad, allowing us to double-buffer our DMA loads. So as our MXU is active in the `k` loop, we are loading the next chunk of B simultaneously, to the inactive B buffer. We will discuss the results of each optimization we do in detail later.

### Optimizing Full Inference

Aside from out matmul, we also implement elementwise operations. This is pretty simple; just copy the tensor(s) to scratchpad, dispatch the VPU on the addresses, and write back to DRAM. If the tensor doesn't fit entirely, load chunks of it into scratchpad and do it multiple times. 

We are now ready to implement a full inference kernel! Using our model architecture, a full prefill, for a single layer, looks like this: 

```
model params: embedding dim d, # heads (heads), head dim h = d/heads

inputs (in external memory, repeated for each layer): 
query tokens S, key/value tokens T
attention weights W_Q/K/V[d, d], FFN weights W_1[d, 4d], W_2[4d, d].
token embeddings X[T, d].

outputs (in external memory, repeated for each layer): 
KV cache [T, 2d].

for each layer: 
    Q[S, d] = tpu_matmul(X, W_Q)
    K[T, d] = tpu_matmul(X, W_K) // directly at respective KV cache address
    V[T, d] = tpu_matmul(X, W_V)  // 
    O[S, d] = empty vector

    for h from 0 to heads-1:
        P[S, T] = tpu_matmul.transposed(Q[:][h * d_h +: d_h], K[h * d_h +: d_h][:])
        P = P + mask
        P = relu(P)
        O[:][h * d_h +: d_h] = P @ V[:][h * d_h +: d_h]
    O = tpu_matmul(O, W_O)
    O = O + X
    O = hardtanh(O)

    A[S, 4d] = tpu_matmul(O, W_1)
    A = relu(A)
    Y[S, d] = tpu_matmul(A, W_2)
    Y = Y + X
    X = hardtanh(Y)
```

In prefill, S=T. Decode is just a special case of this, where S=1, and the K/V matrices are loaded from the KV cache instead of being recomputed every step. Let's run a full inference, with prefill and decode: 

```
base:
run                                    57695827 clocks  4807.986 ms @ 12 MHz
MXU busy                                9752832   16.9%
VPU busy                                 483616    0.8%
DMA busy                               32695968   56.7%
no unit busy (issue overhead)          14763411   25.6%
producer stalled on a full queue              0    0.0%
two or more units busy                        0    0.0%

run                                    47183187 clocks  3931.932 ms @ 12 MHz
MXU busy                                9752832   20.7%
VPU busy                                 483616    1.0%
DMA busy                               32695968   69.3%
no unit busy (issue overhead)          12714599   26.9%
producer stalled on a full queue              0    0.0%
two or more units busy                  8463828   17.9%
```
And with just prefill: 

```
base:
run                                     3719516 clocks  309.960 ms @ 12 MHz
MXU busy                                1097010   29.5%
VPU busy                                 242200    6.5%
DMA busy                                2016984   54.2%
no unit busy (issue overhead)            363322    9.8%
producer stalled on a full queue              0    0.0%
two or more units busy                        0    0.0%

run                                     3372036 clocks  281.003 ms @ 12 MHz
MXU busy                                1097010   32.5%
VPU busy                                 242200    7.2%
DMA busy                                2016984   59.8%
no unit busy (issue overhead)            383315   11.4%
producer stalled on a full queue              0    0.0%
two or more units busy                   367473   10.9%
```
We see that the double buffering helps a lot! it improves our runtime by about 22% in full inference and 10% in prefill. Obviously, we haven't achieved full overlap between compute and comms, but this is probably about as good as I could achieve without designing out-of-order execution in hardware.

But the main observation comes from comparing the prefill and decode. As you can see, decode takes up more than 90% of the actual LLM inference time, which is pretty similar to actual workloads. Furthermore, we see that while prefill's DMA clocks is only about 70% more than MXU (this is actually pretty bad, we will talk more about this later), it is about 3-4x higher in decode. You've probably heard that prefill is compute bound, while decode is memory-bound. We've just shown that here! 

To see why in more detail, let's do some quick math. Let's approximate our transformer as only the FFN blocks, for each layer. For each layer, we load our weight matrices of size $[d, 4d]$ and $[4d, d]$, and our activations $[S,d], [S,4d]$ which are both read and written. On the compute side, we do 2 matmuls: $[S, d]\times [d, 4d]$ and $[S, 4d]\times [4d, d]$. So in total, we are loading $8d^2/2 + 5Sd/2$ clocks, writing back $5Sd/2$ bytes (recalling that our external memory is 1 clock/byte read and 2 clock/byte write, $4d^2 + \frac{15Sd}{2}$ clocks total), and doing $8Sd^2$ operations. 

In prefill, $S=T=64$ for this benchmark. That's $(4)(128)^2=65536$ clocks for weight loads in a layer, $61440$ clocks for activations, and $131072$ clocks of matmul ($8\cdot 6\cdot 128^2$, divided by 64 ops / clock in our MXU). Very comparable matmul and DMA clocks!

Decode is a different story. With our non-batched decode, $S=1$. We still use the same $4(128)^2=65536$ clocks for weight loads. For activations, we only do $\frac{15\cdot 1\cdot 128}{2}=960$ clocks, and we only do $\frac{8\cdot 1\cdot 128^2}{64} = 2048$ clocks of matmul! We've just derived that the comms of decode is completely dominated by weight loads (actually, this changes when the KV cache gets very big, but we won't go into that), and the compute workload is very, very low. And that's what's going on in our TPU.

### Optimizing Prefill

With decode, there's not much room for improvement on our TPU. In fact, if you go through every single weight load in our decode kernel, you'll find that the expected number of clocks just for weights comes very close to the current number. So instead, we are going to focus on optimizing prefill. The flow of prefill is actually very similar to LLM training (with the difference of not having to store every activation, and no backprop), so there is still value in doing this!

#### Operator Fusion

Our current inference code (refer to the pseudocode) has a very obvious improvement we can do: operator fusion. Currently, we are doing a matmul, storing the result back, then loading the result back to scratchpad to do an elementwise add, relu, or hardtanh on it. This makes no sense! Instead, for each output tile in the matmul, we should just do the add/relu/hardtanh before we write back the tensor. We make a new fused matmul function that does this.

Here's the result after implementing fusion: 
```  
run                                     2562124 clocks  213.510 ms @ 12 MHz
MXU busy                                1097010   42.8%
VPU busy                                 242632    9.5%
DMA busy                                1261736   49.2%
no unit busy (issue overhead)            321583   12.6%
producer stalled on a full queue              0    0.0%
two or more units busy                   360837   14.1%
```

Wow! That's a 32% speedup over the double-buffered code, and a 45% speedup over the base prefill!

#### FlashAttention (kinda)

The current benchmark is a bit weird. We have an embedding dim of $128$, but our prefill size is quite small; only $64$ tokens. In real models like DeepSeek-v4-pro, the ratio is very different: while it has a $7168$ embedding dim, it easily support up to 100k+ tokens! So for this next benchmark, we will be increasing the prefill size to $256$. Since our memory is limited, we will be reducing our model to just 1 layer.

But we're not done yet. To see what we should optimize, we need a full picture of every single tensor our inference loads.

Outside of attention, we write and read $K,Q,V,O$ once each. So this amounts to $1.5\cdot 4\cdot Td$ DMA clocks for all of these. Recall that the $1.5$ is because each activation is int4, and the external memory takes 2 clocks to write a byte and 1 clock to read a byte. Then, in the FFN, we read and write a $[T, 4d]$ up-projection tensor and a $[T, d]$ tensor, which is $1.5\cdot 5 \cdot Td$. 

In attention, we calculate our attention scores $P$ of size $[T,T]$. We then multiply that by $V$ to get our attention head output with size $[T, d_h]$. We do this for every head, and each one is both loaded and written (the attention head output is read when multiplying by $W_O$ after concatenating). So our total is $1.5h(T^2 + Td_h) = 1.5T^2h+1.5Td$.

For weights, our $W_{Q/K/V/O}$ weights are each $[d, d]$, so $4d^2 / 2$ bytes. We also have the FFN weights which are $[d, 4d]$ and $[4d, d]$, so $8d^2 / 2$ bytes. In total, we are loading $6d^2$ bytes. We don't have to write them back, so this is $6d^2$ DMA clocks.

Summing it all up: $1.5\cdot 10Td + 1.5T^2h + 6d^2$ DMA clocks. That $T^2$ term is really scary; as our context window grows larger, that will be the majority of the cost. Here's a prefill benchmark on our new architecture: 

```
run                                     2995837 clocks  249.653 ms @ 12 MHz
MXU busy                                1279282   42.7%
VPU busy                                 373848   12.5%
DMA busy                                1265400   42.2%
no unit busy (issue overhead)            207375    6.9%
producer stalled on a full queue              0    0.0%
two or more units busy                   130068    4.3%
```

And let's calculate the cost of the attention score matrix. $1.5\cdot 256^2\cdot 4$ = 393,216. $25%$ of the total comms! And if we reduce our model even more to $d=64$ with prefill size $512$, that grows to nearly $50%$.


## Conclusion

