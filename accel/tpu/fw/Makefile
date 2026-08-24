# Firmware for the PicoRV32 command producer (rtl/cpu_subsys.sv).
#
#   make                  build matmul.hex (the default kernel)
#   make PROG=foo         build foo.hex from foo.c
#   make dis              disassemble the ELF
#   make run PORT=COM5    build, then load + run + check on the board
#   make trace            build + run the kernel NATIVELY, printing its command
#                         trace (docs/picorv32_migration.md §8.1). Needs only the
#                         host cc — no RISC-V toolchain, no board, no simulator.
#   make clean
#
# Needs a bare-metal RISC-V gcc on PATH; the prefix is autodetected the same way
# on Windows, Linux and macOS, override it with CROSS= (e.g.
# CROSS=riscv-none-elf-). `make trace` is the exception: it uses HOSTCC and
# works with no cross toolchain installed at all. CROSS, HOSTCC and PYTHON are
# the three tools this file goes looking for, and each is a plain override.
#
# -march MUST match how cpu_subsys.sv builds the core: COMPRESSED_ISA(1) and
# ENABLE_FAST_MUL(1) but ENABLE_DIV(0), so `mul` is fine and `div`/`rem` would
# decode as an illegal instruction and trap. That is what rv32ic_zmmul says.
# A gcc older than 12 has no zmmul — use -march=rv32ic there and keep division
# and multiplication out of the kernel (libgcc is not linked).

# ---- host portability -------------------------------------------------------
# GNU make runs both its recipes and its $(shell ...) probes through cmd.exe
# whenever no sh is on PATH, which is the normal case on Windows, so nothing
# below may assume a POSIX shell. This is the probe: cmd's echo keeps the quotes
# it was given, every POSIX shell strips them.
ifeq ($(shell echo "probe"),"probe")
WINSHELL := 1
endif

ifdef WINSHELL
which    = $(shell where $(1) 2>NUL)
RMF     := del /q /f
strquote = \"$(1)\"
else
which    = $(shell command -v $(1) 2>/dev/null)
RMF     := rm -f
strquote = '"$(1)"'
endif

# python3 is the name that exists on mac and Linux, python the one that exists
# on Windows; neither is probed when the caller names one.
ifeq ($(origin PYTHON),undefined)
PYTHON := $(if $(WINSHELL),python,$(if $(call which,python3),python3,python))
endif

GOALS := $(or $(MAKECMDGOALS),all)

CROSS_PREFIXES := riscv-none-elf- riscv64-elf- riscv64-unknown-elf- riscv32-unknown-elf-
ifeq ($(origin CROSS),undefined)
CROSS := $(firstword $(foreach p,$(CROSS_PREFIXES),$(if $(call which,$(p)gcc),$(p))))
endif

ifeq ($(CROSS),)
# `trace` and `clean` are host-only, so neither may hard-fail on a missing cross
# compiler — that is the point of the native trace path.
ifneq ($(filter-out clean trace %.trace,$(GOALS)),)
$(error no RISC-V cross gcc found on PATH — install one (brew install riscv64-elf-gcc, \
apt install gcc-riscv64-unknown-elf, or the xPack riscv-none-elf-gcc, which is the \
same toolchain on all three OSes) or pass CROSS=<prefix>-)
endif
endif

CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
OBJDUMP := $(CROSS)objdump
SIZE    := $(CROSS)size

ARCH    := -march=rv32ic_zmmul -mabi=ilp32
CFLAGS  := $(ARCH) -Os -ffreestanding -nostdlib -fno-builtin -Wall -Wextra \
           -ffunction-sections -fdata-sections -MMD -MP

# Problem shape. Empty = the kernel's own #defines. Make does not track a
# changed -D, so `make clean` first when sweeping (../tb/run_fw_sweep.sh does)
# or when RQ changes (../host/run_adder.py deletes the kernel's objects itself).
#   make M=8 KTILES=16 NTILES=16
#   make PROG=infer GEN=3     tokens for infer.c to generate (default T-PROMPT)
#   make PROG=adder RQ=/tmp/adder_rq_ckpt.h
#                             adder.c's requant table, derived from a real
#                             checkpoint by `accel/tpulang/adder_export.py
#                             --emit-rq`, in place of the checked-in adder_rq.h.
#                             The {m0,n} words are literals in the commands, so
#                             an image built without this one carries the
#                             synthetic table and scores nothing.
M      ?=
KTILES ?=
NTILES ?=
GEN    ?=
RQ     ?=
SHAPEFLAGS := $(if $(M),-DM=$(M)) $(if $(KTILES),-DKTILES=$(KTILES)) \
              $(if $(NTILES),-DNTILES=$(NTILES)) $(if $(GEN),-DINFER_GEN=$(GEN)) \
              $(if $(RQ),-DADDER_RQ_H=$(call strquote,$(abspath $(RQ))))
CFLAGS += $(SHAPEFLAGS)
LDFLAGS := $(ARCH) -nostdlib -Wl,-T,link.ld -Wl,--gc-sections

PROG ?= matmul
# memops.o is the freestanding runtime gcc calls without being asked (struct
# assignment lowers to memcpy). --gc-sections drops it from kernels that make
# no such call, so it costs nothing to carry.
OBJS := start.o memops.o $(PROG).o

PORT ?=
PORTARG := $(if $(PORT),-p $(PORT))

.PHONY: all dis size run trace clean
.PRECIOUS: %.elf %.bin %.trace

all: $(PROG).hex

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S
	$(CC) $(CFLAGS) -c -o $@ $<

$(PROG).elf: $(OBJS) link.ld
	$(CC) $(LDFLAGS) -Wl,-Map,$(@:.elf=.map) -o $@ $(OBJS)
	$(SIZE) $@

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

%.hex: %.bin
	$(PYTHON) bin2hex.py $< $@

dis: $(PROG).elf
	$(OBJDUMP) -d $<

size: $(PROG).elf
	$(SIZE) -A $<

# The host end: load the firmware over 'I', stage the operands, run, read back.
run: $(PROG).hex
	$(PYTHON) ../host/run_fw_matmul.py --fw $(PROG).hex $(PORTARG)

# ---- native command trace (docs/picorv32_migration.md §8.1) ------------------
# The same .c, the same builders out of tpu.h, compiled by the host compiler
# with the two MMIO primitives swapped for a trace emitter. This is the producer
# that replaces the RV32IM interpreter, and the reason it cannot drift from the
# firmware is that nothing but those two primitives differs between the builds.
# `cc` is the POSIX spelling and is what a mac or Linux box has; a Windows box
# usually has none of these, so a missing host compiler is only an error once a
# trace is actually the goal.
ifeq ($(origin HOSTCC),undefined)
HOSTCC := $(firstword $(foreach c,cc gcc clang,$(if $(call which,$(c)),$(c))))
endif
ifeq ($(HOSTCC),)
ifneq ($(filter trace %.trace,$(GOALS)),)
$(error no host C compiler found — install one or pass HOSTCC=<compiler>)
endif
endif
TRACEFLAGS := -DTPU_TRACE -I. -Wall -Wextra -O1 $(SHAPEFLAGS)

# The host build gets memcpy from libc, so memops.c is not in this line.
%.trace: %.c mock/tpu_trace.c tpu.h tpulib.h
	$(HOSTCC) $(TRACEFLAGS) -o $@ $< mock/tpu_trace.c

# stdin closed on purpose. A kernel that reads the scratchpad window blocks for
# an answer here (mock/tpu_trace.c), and there is nobody to give it one: this
# target prints a program's commands, it does not run its arithmetic. At EOF the
# read returns 0 with a warning, so `make trace PROG=infer` shows the shape of
# the kernel instead of hanging. The trace that *means* something comes from
# fw_vectors.py -x, which runs this same binary with the ISS on the other end.
#
# cmd.exe refuses to exec a file whose extension is not in PATHEXT, and the
# binary has to keep the .trace name the rest of the toolchain passes around, so
# on Windows python launches it — CreateProcess does not care about the
# extension — and closes its stdin the way `< /dev/null` does.
ifdef WINSHELL
RUNTRACE = $(PYTHON) -c "import subprocess,sys; sys.exit(subprocess.run([sys.argv[1]], stdin=subprocess.DEVNULL).returncode)" $(PROG).trace
else
RUNTRACE = ./$(PROG).trace < /dev/null
endif

trace: $(PROG).trace
	$(RUNTRACE)

clean:
	-$(RMF) *.o *.d *.elf *.bin *.hex *.map *.trace

-include $(OBJS:.o=.d)
