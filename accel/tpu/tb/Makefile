# Unit testbenches for accel/tpu's RTL blocks (Icarus Verilog, -g2012).
#
# Each test <name> pairs tb/<name>_tb.sv with rtl/<name>.sv. Pick one with TEST:
#
#   make TEST=mxu           compile + run mxu_tb
#   make TEST=vpu sim       same (sim is the default target)
#   make TEST=mxu wave      run and remind you where the VCD is (tb/mxu_tb.vcd)
#   make list               show every test that has a tb/<name>_tb.sv
#   make all                run every discovered test in turn
#   make clean              remove all *_tb.vvp / *_tb.vcd
#
# Multi-file tests: override RTL to pass extra sources, e.g.
#   make TEST=mxu RTL="../rtl/mxu.sv ../rtl/requant.sv"
#
# Three aliases for the board-image cores:
#   make echo    the UART block loopback (docs/uart_selftest.md)
#   make mem     UART + external SRAM, no TPU core
#   make bram    the same protocol against on-chip block RAM
#
# WHOLE-CORE tests are not here. fw_matmul_tb.sv and fw_uart_tb.sv run a
# firmware kernel through the entire core against golden vectors, and both are
# driven by accel/test (RTLBackend), which builds the image, generates the
# vectors and invokes iverilog itself:
#
#   python accel/test/run_suite.py -b rtl
#   python accel/test/tests/matmul/generate.py -b rtl
#
# core.f is the file list for the whole core, shared with that driver.

TEST ?= mxu

TOP := $(TEST)_tb
TB  := $(TEST)_tb.sv

# Tests needing more than rtl/<name>.sv declare it here as RTL_<name>, so
# `make TEST=<name>` works without spelling the list out on the command line.
# An explicit RTL="..." on the command line still wins over both.
RTL_uart_echo := ../rtl/uart_echo.sv ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv
RTL_uart_interface := ../rtl/uart_interface.sv ../rtl/uart_receiver.sv \
                      ../rtl/uart_transmitter.sv ../rtl/sram.sv
RTL_uart_memory := ../rtl/uart_memory.sv ../rtl/uart_interface.sv \
                   ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv ../rtl/sram.sv
RTL_uart_bram := ../rtl/uart_bram.sv ../rtl/uart_interface.sv \
                 ../rtl/uart_receiver.sv ../rtl/uart_transmitter.sv ../rtl/bram.sv
# dma_tb instantiates the DUT against the real scratchpad and a behavioral SRAM
# chip. The DMA drives the chip pins itself, so sram.sv is not in the list.
RTL_dma := ../rtl/dma.sv ../rtl/scratchpad.sv
# The command front ends share one FIFO module.
RTL_cmd_queue := ../rtl/cmd_queue.sv
RTL_cmd_plane := ../rtl/cmd_queue.sv ../rtl/cmd_mxu.sv ../rtl/cmd_vpu.sv ../rtl/cmd_dma.sv
# cpu_smoke and tpu_top need the whole core; core.f is that list.
RTL_cpu_smoke := -f core.f
RTL_tpu_top   := -f core.f

RTL ?= $(if $(RTL_$(TEST)),$(RTL_$(TEST)),../rtl/$(TEST).sv)

# Extra iverilog flags, for tests that take a -D switch:
#   make TEST=sram IVFLAGS=-DCPA=2      the SRAM controller's slower beat
IVFLAGS  ?=
IVERILOG := iverilog -g2012 -Wall $(IVFLAGS)
VVP      := vvp

# fw_matmul / fw_uart are excluded: they need a firmware image and golden
# vectors, which accel/test builds. `make all` would only fail on them.
TESTS := $(filter-out fw_matmul fw_uart,$(patsubst %_tb.sv,%,$(wildcard *_tb.sv)))

.PHONY: sim wave list all clean echo mem bram

sim: $(TB)
	$(IVERILOG) -o $(TOP).vvp $(RTL) $(TB)
	$(VVP) $(TOP).vvp

echo:
	@$(MAKE) --no-print-directory TEST=uart_echo sim

mem:
	@$(MAKE) --no-print-directory TEST=uart_memory sim

bram:
	@$(MAKE) --no-print-directory TEST=uart_bram sim

wave: sim
	@echo "open $(TOP).vcd in gtkwave"

list:
	@echo "available tests:"
	@for t in $(TESTS); do echo "  make TEST=$$t"; done
	@echo "whole-core kernels: python accel/test/run_suite.py -b rtl"

all:
	@for t in $(TESTS); do \
		echo "=== running $$t ==="; \
		$(MAKE) --no-print-directory TEST=$$t sim || exit $$?; \
	done

clean:
	rm -f *_tb.vvp *_tb.vcd
