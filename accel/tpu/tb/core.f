# The whole core, as one iverilog file list. Used by this directory's Makefile
# and by accel/test/backends.py, so the list is written once. Paths are
# relative to tb/, which is where both callers run iverilog from.
../rtl/tpu_top.sv
../rtl/mxu.sv
../rtl/vpu.sv
../rtl/scratchpad.sv
../rtl/dma.sv
../rtl/perf_counters.sv
../rtl/cmd_queue.sv
../rtl/cmd_mxu.sv
../rtl/cmd_vpu.sv
../rtl/cmd_dma.sv
../rtl/cpu_subsys.sv
../rtl/vendor/picorv32.v
../rtl/uart_interface.sv
../rtl/uart_receiver.sv
../rtl/uart_transmitter.sv
