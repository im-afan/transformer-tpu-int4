# Firmware for the PicoRV32 command producer (rtl/cpu_subsys.sv).
#
# This directory is the library — tpu.h, tpulib.h, start.S, memops.c, link.ld.
# The kernels themselves live with their vectors in accel/test/tests/<name>/,
# and accel/test/backends.py is what normally invokes this:
#
#   make PROG=infer SRC=/abs/path/infer.c BUILD=/abs/build EXTRA_CFLAGS="-I... -DX=1"
#
# By hand it works the same way; SRC defaults to $(PROG).c in this directory and
# BUILD to `.`, so `make PROG=foo` still builds foo.hex here if a foo.c is here.
#
#   make dis PROG=infer BUILD=...   disassemble the ELF
#   make size PROG=infer BUILD=...  section sizes
#   make clean BUILD=...
#
# Needs a bare-metal RISC-V gcc on PATH; the prefix is autodetected the same way
# on Windows, Linux and macOS, or pass CROSS= (e.g. CROSS=riscv-none-elf-).
#
# -march MUST match how cpu_subsys.sv builds the core: COMPRESSED_ISA(1) and
# ENABLE_FAST_MUL(1) but ENABLE_DIV(0), so `mul` is fine and `div`/`rem` would
# decode as an illegal instruction and trap. That is what rv32ic_zmmul says.
# A gcc older than 12 has no zmmul — use -march=rv32ic there and keep division
# out of the kernel (libgcc is not linked).

# ---- host portability -------------------------------------------------------
# GNU make runs recipes and $(shell ...) probes through cmd.exe whenever no sh
# is on PATH, so nothing below may assume a POSIX shell. This is the probe:
# cmd's echo keeps the quotes it was given, every POSIX shell strips them.
ifeq ($(shell echo "probe"),"probe")
WINSHELL := 1
endif

ifdef WINSHELL
which = $(shell where $(1) 2>NUL)
RMF  := del /q /f
else
which = $(shell command -v $(1) 2>/dev/null)
RMF  := rm -f
endif

ifeq ($(origin PYTHON),undefined)
PYTHON := $(if $(WINSHELL),python,$(if $(call which,python3),python3,python))
endif

GOALS := $(or $(MAKECMDGOALS),all)

CROSS_PREFIXES := riscv-none-elf- riscv64-elf- riscv64-unknown-elf- riscv32-unknown-elf-
ifeq ($(origin CROSS),undefined)
CROSS := $(firstword $(foreach p,$(CROSS_PREFIXES),$(if $(call which,$(p)gcc),$(p))))
endif
ifeq ($(CROSS),)
ifneq ($(filter-out clean,$(GOALS)),)
$(error no RISC-V cross gcc found on PATH — install one (brew install riscv64-elf-gcc, \
apt install gcc-riscv64-unknown-elf, or the xPack riscv-none-elf-gcc, which is the \
same toolchain on all three OSes) or pass CROSS=<prefix>-)
endif
endif

CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
OBJDUMP := $(CROSS)objdump
SIZE    := $(CROSS)size

FWDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ARCH  := -march=rv32ic_zmmul -mabi=ilp32

# -O2, not -Os: tpulib.h's block arithmetic divides by the shape, and at -Os gcc
# emits a __udivsi3 call for that instead of folding it. There is no libgcc for
# this multilib and the link is -nostdlib, so a kernel does not link at -Os.
OPT ?= -O2
EXTRA_CFLAGS ?=
CFLAGS := $(ARCH) $(OPT) -ffreestanding -nostdlib -fno-builtin -Wall -Wextra \
          -ffunction-sections -fdata-sections -MMD -MP -I$(FWDIR) $(EXTRA_CFLAGS)
LDFLAGS := $(ARCH) -nostdlib -Wl,-T,$(FWDIR)link.ld -Wl,--gc-sections

PROG  ?= infer
SRC   ?= $(FWDIR)$(PROG).c
BUILD ?= .

# memops.o is the freestanding runtime gcc calls without being asked (struct
# assignment lowers to memcpy). --gc-sections drops it from kernels that make
# no such call, so it costs nothing to carry.
OBJS := $(BUILD)/start.o $(BUILD)/memops.o $(BUILD)/$(PROG).o

.PHONY: all dis size clean
.PRECIOUS: $(BUILD)/%.elf $(BUILD)/%.bin

all: $(BUILD)/$(PROG).hex

# The kernel, wherever it lives. Its own directory is on the include path so a
# generated config header can sit next to it.
$(BUILD)/$(PROG).o: $(SRC)
	$(CC) $(CFLAGS) -I$(dir $(SRC)) -c -o $@ $<

$(BUILD)/%.o: $(FWDIR)%.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: $(FWDIR)%.S
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/$(PROG).elf: $(OBJS) $(FWDIR)link.ld
	$(CC) $(LDFLAGS) -Wl,-Map,$(@:.elf=.map) -o $@ $(OBJS)
	$(SIZE) $@

$(BUILD)/%.bin: $(BUILD)/%.elf
	$(OBJCOPY) -O binary $< $@

$(BUILD)/%.hex: $(BUILD)/%.bin
	$(PYTHON) $(FWDIR)bin2hex.py $< $@

dis: $(BUILD)/$(PROG).elf
	$(OBJDUMP) -d $<

size: $(BUILD)/$(PROG).elf
	$(SIZE) -A $<

clean:
	$(RMF) $(BUILD)/*.o $(BUILD)/*.d $(BUILD)/*.elf $(BUILD)/*.bin \
	       $(BUILD)/*.hex $(BUILD)/*.map

-include $(OBJS:.o=.d)
