# Software rewrite

## Purpose: for better readability, cleanliness, and easy to scale to more archtictures and a full firmware test suite

## Flow

### Verification Suite
- backends.py 
    - defines Backend class, 2 inheritors: TPUBackend, ISSBackend
        - def run(fw (path to c file), input_dram (path to input DRAM .hex))
            - output: output dram .hex string
            - compiles the fw/ file and runs it on either TPUBackend or ISSBackend,
            - reads the result from the backend DRAM and returns it
        - ISSBackend: 
            - calls ISS class in iss.py, no changes needed
            - coexecutes with compiled fw c file, similar to how it is right now
        - TPUBackend: 
            - loads compiled fw c file (using riscv64-unknown-elf-gcc etc.) & input dram to TPU, sends RUN signal, reads it back, using tpu_uart.py
            - tpu_uart.py changes: it's very spaghetti right now, get rid of everything except for basic functions (read dram, write dram, write instr, go, read timer)
- vector_generator.py: defines VectorGenerator class
    - VectorGenerator: def generate_vectors() -> input .hex, output .hex
- program.py: defines TPUProgram class
    - attributes: c_source (c file), backend (ISSBackend or TPUBackend), generate_vectors (VectorGenerator class)
        - generate_vectors generates input & golden output .hex file 
    - def run_program()
        - runs program using backend, with input generated from generate_vectors -> get output .hex
        - compare output .hex with generate_vectors .hex
    - def read_timers()
        - TPUBackend only; read timers

- example usage: test matmul
    - have all the test c & python sources together in a single folder 
    - c source: standard firmware file
    - python source: 
        - instantiate VectorGenerator with torch matmul to write expected dram out
        - instantiate TPUProgram with c source, backend, VectorGenerator
    - TPUProgram.run_program() -> get whether they match 
- make a test suite with multiple such tests

### LLM Inference
- numbers_data.py, train.py, transformer.py: training; defines architecture,
    - GOAL: export model.pt, no changes needed
- export.py: exports transformer architecture model.pt to an input.hex and one infer.c file
    - flexible based on architecture shapes (L, d, d_ff, batch, tokens, etc)
        - by changing defines in infer.c, including dram addresses
        - fills in requant table
    - export steps: 
        - load .pt
        - export weights by iterating through modules, always the same order in DRAM, but addresses can change based on sizes
            - iteration hardcoded using transformer.py arch
            - out: input .hex file
            - fill in addresses by changing infer.c defines, result: matching infer.c with model shape
        - calculate requant values, control flow also hardcoded by transformer.py arch
            - fill in requant table

- running a model: 
    - similar to test matmul example
        - instantiate TPUProgram with c source, backend, dummy VectorGenerator (generates just input .hex with .hex and token inputs)
        - TPUProgram.run_program()
        - read output .hex tokens