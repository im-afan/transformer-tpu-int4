# Accelerators

Hardware/backend implementations of the operations in [`../model`](../model).

- `tpu/`: Custom SystemVerilog TPU on a Digilent Cmod A7-35T: the RTL, the firmware library, the block testbenches and the Vivado build
- `test/`: The verification suite: the bit-exact ISS, the three backends, the vector generators, the checkpoint exporter

## `test/`: kernel testing with 3 backends, checked against NumPy baseline 

A kernel is a `.c` file and a `generate.py` next to it. The generator writes the operands
and the expected answer; `ISSBackend`, `RTLBackend` and `TPUBackend` are three
interchangeable ways of producing an answer to compare against.

```bash
python accel/test/run_suite.py           # every kernel, on the ISS
python accel/test/run_suite.py -b rtl    # ...through the whole core in Icarus
python accel/test/run_suite.py -b board -p /dev/ttyUSB1
```