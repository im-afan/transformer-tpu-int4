# Transformer TPU writeup

## Introduction

This was a summer project I made to learn the basics of ML systems. This article will be split into different parts, so feel free to scroll through! The goal of this project is to derive some of the common results you'll find in ML (limited to a single device for now), and to get a feel for the math behind model scaling. I'm assuming you have basic knowledge on how transformer inference works (prefill, decode), but not the hardware side of things.

## What determines how fast our model is?

Before we get into hardware, we need to ask a seemingly obvious question: what makes a model/algorithm faster? In ML, this problem can be analyzed in 2 domains: compute and communications. Compute is how many raw operations we can do in some amount of time. For example, FLOP/s (floating point operations / sec) is a common metric for the compute capabilities of GPUs or other ML accelerators. So the total compute time of an algorithm can be calculated using $T_compute = # FLOPs / accelerator FLOPs/s$. 

On the other hand, we have communication. In single-device inference, this usually refers to the comms between the memory (HBM, DDR) and the accelerator's cache. In distributed inference, the comms between devices must also be considered. Similarly, if we know the memory bandwidth of our HBM or DDR, we can calculate our comms time as $T_comms = Communication Bytes / Memory Bytes/sec$




