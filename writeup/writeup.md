# Transformers, TPUs, Kernels, and Lessons About AI-Assisted Development 

## Introduction

This was a summer project I made to learn the basics of ML systems. This article will be split into different parts, so feel free to scroll through! The goal of this project is to derive some of the common results you'll find in ML (limited to a single device for now), and to get a feel for the math behind model scaling. I'm assuming you have basic knowledge on how transformer inference works (prefill, decode), but not the hardware side of things.

## Background

### What determines how fast our model is?

Before we get into hardware, we need to ask a seemingly obvious question: what makes a model/algorithm faster? In ML, this problem can be analyzed in 2 domains: compute and communications. Compute is how many raw operations we can do in some amount of time. For example, FLOP/s (floating point operations / sec) is a common metric for the compute capabilities of GPUs or other ML accelerators. So the total compute time of an algorithm can be calculated using $T_compute = # FLOPs / accelerator FLOPs/s$. 

On the other hand, we have communication. In single-device inference, this usually refers to the comms between the memory (HBM, DDR) and the accelerator's cache. In distributed inference, the comms between devices must also be considered. Similarly, if we know the memory bandwidth of our HBM or DDR, we can calculate our comms time as $T_comms = Communication Bytes / Memory Bytes/sec$

In most hardware, we assume that comms and compute run at the same time, so optimally, they are completely overlapped. So, our lower bound on the runtime of an algorithm is $max(T_comms, T_compute)$. 

### Matmuls

Matmuls have a property regarding compute and comms that makes them so special for ML. First, let us consider an $N\times N\times N$ matmul, in int8: $A[N, N] \cdot B[N, N] = C[N, N]$. In the best case, we have to load $2N^2$ bytes to our compute, then write back $N$ bytes to memory, which is a total of $3N^2$. For compute, we have to perform $N^3$ int8 multiplications. So the total arithmetic intensity of the algorithm is on the order of N, meaning it is more compute-demanding the larger our matmul gets. This is partly why matmuls are so essential to ML workloads: it is very easy to scale up our models by just throwing more compute at it to achieve larger matmuls.  


## Model Architecture / Goal

The overall goal is to run a transformer model on our design, while not making it completely fixed to a single architecture. We want to be able to write code to change the model architecture/dimensions, tweak our matmuls to be more efficient, or run unit tests to benchmark single layers or matmuls. This means that having fixed control flow set in hardware is not acceptable.

We will train and inference our model on predicting the next token in an addition sequence with a maxmium of 31 digits per input (64 token prefill and 64 token decode overall), which is a pretty simple but nontrivial task. 

For benchmarking, we use a standard Transformer architecture, with embedding dim $d=128$, ffn dim $d_{ff}=512$, 4 QKV heads, and 4 layers. However, we greatly simplify some operations to allow for ease of implementation. First, we replace gelu with relu activation. Softmax in attention is also replaced with a simple ReLU, which has a big effect on training but is fine for our task. LayerNorm is replaced with a tanh normalization. Finally, we train with no bias in the ffn. While these are pretty major simplifications, the overall architecture stays the same, and the original goal of analyzing transformer inference performance can still be achieved.

## Hardware

### Overview

In a transformer, we have 2 types of operations: matmuls, and elementwise ops like tensor addition and ReLU. The purpose of an accelerator is to load tensors from memory, do those operations, and write back the results.

The memory heirarchy of our TPU is simple. We are using a Cmod A7 board, which includes an Artix-7 FPGA chip along with an external asynchronous SRAM chip (8 bit read, 8 ns access time). We use the SRAM chip to model our accelerator's external memory (HBM/DDR in a real accelerator), and the Artix-7's BRAM to act as an on-chip cache (scratchpad memory). 

There are 3 main units: MXU, VPU, and DMA. The MXU (matrix mutliply unit) handles the matrix multiplications, VPU (vector processing unit) handles vector operations such as activations and addition. Both units read from the scratchpad memory and write their results back to scratchpad. The DMA (direct memory access) handles transfers between scratchpad and external memory. Each unit is controlled by a softcore PicoRV32 processor, which we can write C firmware to dispatch instructions to each unit through AXI-based MMIO.

[TPU ARCHITECTURE DIAGRAM]

### Memory 

Our scratchpad memory is synthesized as simple dual-port BRAM, with 2 independent ports: read and write. Since it can be synthesized to basically an arbitrary bus width for our usage (1 x 32K bits to 72 x 512 bits), we don't need the blocks to be traditional banks by `address % bus width`. Instead, we can use a single block to represent a contiguous region of our scratchpad memory. This basically allows us to have as many ports accessing scratchpad as we want, with the limitation that 2 ports don't access the same region; this will be very useful in the design later.

[BRAM DIAGRAM]

Our DMA is very simple. Running at the Cmod A7's standard 12 MHz clock, the 8ns external RAM access time is basically instant; it arrives at the next clock. The DMA unit takes in a base address for scratchpad and external RAM, a row stride, and 2d matrix dimensions, and simply drives the ports of the external RAM and scratchpad to transfer data between them.


### MXU & VPU

For matmuls, we use an output-stationary 8x8 systolic array; each PE keeps its partial sum, while moving its input values to the next PE to its right and below it. This allows us to perform an arbitrary 8xNx8 ($A[8,N] \cdot B[N,8])$) mamtul, as long as A and B fit in scratchpad. By default, all tensors are stored row-major in both external memory and scratchpad. Since our design has no fast way to transpose a matrix, we instead use a trick in the MXU to perform transposed matmuls such as $QK^T$ in attention. By default, A and B are fed into the systolic array by... [FINISH EXPLANATION, PROBABLY NEED A DIAGRAM FOR THIS TO BE EXPLAINED WELL] 

As a result, every clock, the systolic array reads $4N$ bits for matrix A and $4N$ bits for matrix B. This is where the scratchpad architecture comes in handy: as long as A and B are in different memory regions, they can be read at the same time, without interfering with ongoing DMA operations.

The VPU is pretty simple. Its inputs are the base address of a vector and the length of the operation. It repeatedly loads chunks of the vector from scratchpad, operates on them (add, relu, etc), and writes them back, until the operation is completed.

### Design Choices & Notes

We use PicoRV32 so that we can easily write firmware for different architectures. Not only does this allow for unit testing beyond just inference & different architectures, it allows us to easily experiment with optimizations later on without having to change the dataflow in hardware.

CPU issue overhead was also a concern when designing the architecture. When the CPU dispatches instructions, they enter a command queue for each unit, which are then executed asynchronous of the CPU execution order. For synchronization, the CPU can also poll each instruction queue's state. Furthermore, to minimize the effect of execution latency, the TPU operations are intentionally complex, allowing for instructions to span across large address ranges without needing more CPU executes.

## Firmware & Kernels

### 