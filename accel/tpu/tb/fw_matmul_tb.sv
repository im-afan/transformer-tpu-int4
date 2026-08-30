`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// fw_matmul_tb.sv — the C firmware matmul through the whole core.
//
// cpu_smoke_tb proves the CPU can push a DMA command. This proves it can drive
// the whole core: any kernel in accel/test/tests/ runs here, against goldens
// the same kernel's native build produced on the ISS.
//
// Not run from this directory's Makefile. accel/test's RTLBackend builds the
// image, writes the three vector files, invokes iverilog and reads the result:
//
//   python accel/test/run_suite.py -b rtl
//   python accel/test/tests/matmul/generate.py -b rtl --ktiles 16
//
// The firmware is loaded through cpu_subsys.sv's FW_INIT ($readmemh), not over
// the UART, so this is a backdoor test: seed DRAM directly, pulse host_run with
// the producer bit set, wait for `done`, check DRAM. Nothing drives the serial
// pins — fw_uart_tb.sv is the one that does.
//
//   <vecdir>/fw_dram_in.hex    operands to seed DRAM with
//   <vecdir>/fw_dram_exp.hex   every DRAM byte the run should write
//   <vecdir>/fw_cmds.hex       the command trace the firmware should produce
//   +DRAMOUT=<path>            dump the final SRAM for the Python driver
//
// Two things are therefore checked, not one:
//
//   1. the DRAM image           — did the datapath compute the right answer
//   2. the **command trace**    — did the CPU issue the right commands
//
// (2) is the point of phase 3. Without it a failure says only "wrong answer";
// with it, a diverging command names the producer and a matching trace with a
// wrong image names the datapath.
// -----------------------------------------------------------------------------

`ifndef FW_HEX
  `define FW_HEX "../fw/matmul.hex"
`endif
// Problem shape. Must match what the firmware was built with -- ../fw/Makefile
// takes the same three numbers, and run_fw_sweep.sh passes both from one place.
`ifndef FW_M
  `define FW_M 8
`endif
`ifndef FW_KT
  `define FW_KT 4
`endif
`ifndef FW_NT
  `define FW_NT 2
`endif
`ifndef WATCHDOG_NS
  `define WATCHDOG_NS 2000000
`endif
`ifndef FW_VEC_DIR
  `define FW_VEC_DIR "vectors_fw"
`endif

module fw_matmul_tb;

    localparam int ROWS       = 8;
    localparam int COLS       = 8;
    localparam int ADDR_W     = 16;
    localparam int XLEN       = 32;
    localparam int M0_W       = 12;
    localparam int N_W        = 4;
    localparam int IMEM_AW    = 10;
    localparam int CFG_AW     = 5;
    localparam int FW_AW      = 12;
    localparam int MEM_ADDR_W = 19;
    localparam int MEM_DATA_W = 8;

    localparam int CLK_NS = 10;

    // ---- the problem (mirrors fw/matmul.c) ----------------------------------
    localparam int M      = `FW_M;
    localparam int KTILES = `FW_KT;
    localparam int NTILES = `FW_NT;
    localparam int K      = KTILES * ROWS;
    localparam int N      = NTILES * COLS;

    localparam int AROW = K;                 // A row stride, bytes
    localparam int WROW = (N * 4) / 8;       // W row stride, bytes (row-major int4)
    localparam int CROW = N * 4;             // C row stride, bytes

    // Bases spaced for shapes past the default: A <= 8 KB, W <= 8 KB, C <= 48 KB.
    localparam [MEM_ADDR_W-1:0] A_ADDR = 19'h0_0000;
    localparam [MEM_ADDR_W-1:0] W_ADDR = 19'h0_2000;
    localparam [MEM_ADDR_W-1:0] C_ADDR = 19'h0_4000;
    localparam int              A_BYTES = M * K;
    localparam int              W_BYTES = K * WROW;
    localparam int              C_BYTES = M * N * 4;

    int errors = 0, checks = 0;
    int unsigned cyc = 0, run_clk = 0;

    // Golden images, $readmemh'd from vectors_fw/. `x` means "not written by
    // this run", which is how the expected image stays sparse without needing a
    // separate mask: only the bytes the ISS run recorded are compared.
    // matmul_loop issues 1 + KTILES*NTILES matmuls plus 3 DMAs, so the sweep's
    // largest shape (16x16 tiles) needs 260; `adder` is 534 and `infer` — a
    // whole prefill plus 16 decode steps in one run — is 4606. Overflow is
    // reported, not silently truncated: a capped trace would compare equal on
    // its first MAX_CMDS entries and say nothing about the rest.
    localparam int MAX_CMDS = 8192;
    logic [7:0]  dram_in  [0:(1<<MEM_ADDR_W)-1];
    logic [7:0]  dram_exp [0:(1<<MEM_ADDR_W)-1];
    logic [31:0] cmd_exp  [0:MAX_CMDS*5-1];
    int          n_cmd_exp;

    logic clk = 0, rst_n = 0;
    always #(CLK_NS/2) clk = ~clk;
    always @(posedge clk) cyc++;

    logic                  host_run;
    logic [FW_AW:0]        boot_pc;
    logic                  busy, done;
    logic [IMEM_AW-1:0]    pc_dbg;

    wire  [MEM_ADDR_W-1:0] sram_addr;
    wire  [MEM_DATA_W-1:0] sram_data;
    wire                   sram_we, sram_ce, sram_oen;
    wire                   uart_tx;

    tpu_top #(
        .ROWS(ROWS), .COLS(COLS), .ADDR_W(ADDR_W),
        .XLEN(XLEN), .M0_W(M0_W), .N_W(N_W), .IMEM_AW(IMEM_AW), .CFG_AW(CFG_AW),
        .FW_AW(FW_AW), .FW_INIT(`FW_HEX),
        .MEM_STYLE("BRAM"), .MEM_ADDR_W(MEM_ADDR_W), .MEM_DATA_W(MEM_DATA_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .host_run(host_run), .boot_pc(boot_pc),
        .busy(busy), .done(done), .pc_dbg(pc_dbg),
        .imem_we(1'b0), .imem_waddr('0), .imem_wdata('0),
        .cfg_we(1'b0), .cfg_waddr('0), .cfg_wdata('0),
        .sram_addr(sram_addr), .sram_data(sram_data),
        .sram_we(sram_we), .sram_ce(sram_ce), .sram_oen(sram_oen),
        .uart_rx(1'b1), .uart_tx(uart_tx)
    );

    // ---- behavioural async SRAM chip (same model as tpu_top_tb) -------------
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [MEM_DATA_W-1:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (sram_ce == 1'b0) && (sram_oen == 1'b0) && (sram_we == 1'b1);
    assign #3 sram_data = mem_drives ? sram_mem[sram_addr] : {MEM_DATA_W{1'bz}};
    always @(posedge sram_we) if (sram_ce == 1'b0) sram_mem[sram_addr] <= sram_data;

    // ---- golden vectors ------------------------------------------------------
    task automatic load_vectors();
        for (int i = 0; i < SRAM_SZ; i++) begin
            dram_in[i]  = 8'hxx;
            dram_exp[i] = 8'hxx;
        end
        for (int i = 0; i < MAX_CMDS*5; i++) cmd_exp[i] = 32'hxxxx_xxxx;

        $readmemh({`FW_VEC_DIR, "/fw_dram_in.hex"},  dram_in);
        $readmemh({`FW_VEC_DIR, "/fw_dram_exp.hex"}, dram_exp);
        $readmemh({`FW_VEC_DIR, "/fw_cmds.hex"},     cmd_exp);

        if (dram_in[A_ADDR] === 8'hxx) begin
            $display("FW_MATMUL: no vectors in %s - run this through accel/test", `FW_VEC_DIR);
            $fatal(1);
        end

        n_cmd_exp = 0;
        while (n_cmd_exp < MAX_CMDS && cmd_exp[n_cmd_exp*5] !== 32'hffff_ffff &&
               cmd_exp[n_cmd_exp*5] !== 32'hxxxx_xxxx)
            n_cmd_exp++;
    endtask

    task automatic seed_dram();
        for (int i = 0; i < SRAM_SZ; i++)
            sram_mem[i] = (dram_in[i] === 8'hxx) ? 8'h00 : dram_in[i];
    endtask

    // ---- command-trace monitor -----------------------------------------------
    // Hooked on tpu_top's arbitrated command write, which is where both
    // producers converge (tpu_top.sv `p_cmd_*`) — one probe covers the CPU and
    // the scalar unit, and it sees exactly what enters the queues.
    int          n_cmd_got = 0;
    int unsigned cmd_got [0:MAX_CMDS*5-1];

    always @(posedge clk) begin
        if (rst_n && dut.p_cmd_we && !dut.p_cmd_full) begin
            if (n_cmd_got < MAX_CMDS) begin
                cmd_got[n_cmd_got*5 + 0] = dut.p_cmd_unit;
                cmd_got[n_cmd_got*5 + 1] = dut.p_cmd_data[31:0];
                cmd_got[n_cmd_got*5 + 2] = dut.p_cmd_data[63:32];
                cmd_got[n_cmd_got*5 + 3] = dut.p_cmd_data[95:64];
                cmd_got[n_cmd_got*5 + 4] = dut.p_cmd_data[127:96];
            end
            n_cmd_got++;
        end
    end

    // ---- optional per-command timeline (+CMDLOG=<path>) ----------------------
    //
    // `idlec` says how many clocks of the run had no unit busy. On a 534-command
    // kernel that number alone cannot say *where*, and "where" is the whole
    // question — 4% spread evenly is a different machine from 4% concentrated in
    // one phase. This records, per command, the clock the CPU pushed it and the
    // clocks its unit began and finished it, which is enough to attribute every
    // clock of the run to a producer or a consumer.
    //
    // Off unless the plusarg is given, and everything below is in ONE always
    // block on purpose: `cyc` is incremented by its own always block, so probes
    // split across blocks could disagree by a clock about what "now" is.
    //
    // Pairing is exact rather than heuristic. A GEOM command retires in one
    // clock and never starts its unit, so the nth `*_start` pulse belongs to the
    // nth *executing* command pushed to that unit — each unit's queue is
    // in-order and each command runs to completion. tb/cmd_timeline.py consumes
    // the CSV.
    // The opcode that makes a command actually run its unit: MXU_MM, VPU_OP,
    // DMA_MOVE. A function rather than an unpacked localparam array, which
    // Icarus does not accept.
    function automatic logic [7:0] tl_exec_op(input int unit);
        tl_exec_op = (unit == 0) ? 8'h02 : 8'h01;
    endfunction

    integer tl_f = 0;
    string  tl_path;

    int tl_push  [0:MAX_CMDS-1];
    int tl_start [0:MAX_CMDS-1];
    int tl_end   [0:MAX_CMDS-1];
    // Clocks its unit was actually busy on this command, and the no-unit-busy
    // clocks that preceded its push. Accumulated rather than derived from
    // start/end because the three units do not agree on where `busy` sits
    // relative to their start/done pulses -- and these two columns have to sum
    // to the perf counters exactly or the attribution is guesswork.
    int tl_busy  [0:MAX_CMDS-1];
    int tl_gap   [0:MAX_CMDS-1];
    int tl_unit  [0:MAX_CMDS-1];
    int tl_op    [0:MAX_CMDS-1];
    int tl_w0    [0:MAX_CMDS-1];
    int tl_w1    [0:MAX_CMDS-1];
    int tl_w2    [0:MAX_CMDS-1];
    int tl_q     [0:2][0:MAX_CMDS-1];   // per unit: indices of executing commands
    int tl_wr    [0:2];
    int tl_rd    [0:2];
    int tl_cur   [0:2];
    // Its own push counter rather than the trace monitor's `n_cmd_got`: both
    // blocks run on the same edge and their order is not defined, so reading the
    // other block's counter shifted every record by one.
    int tl_n = 0;
    int tl_idle_acc = 0;   // no-unit-busy clocks since the last push

    initial begin
        for (int i = 0; i < MAX_CMDS; i++) begin
            tl_push[i] = -1; tl_start[i] = -1; tl_end[i] = -1;
            tl_busy[i] = 0;  tl_gap[i] = 0;
        end
        for (int u = 0; u < 3; u++) begin
            tl_wr[u] = 0; tl_rd[u] = 0; tl_cur[u] = -1;
        end
        if ($value$plusargs("CMDLOG=%s", tl_path)) begin
            tl_f = $fopen(tl_path, "w");
            if (tl_f == 0) $display("FW_MATMUL: cannot open %s for the timeline", tl_path);
        end
    end

    always @(posedge clk) if (rst_n && tl_f != 0) begin
        int u;
        if (dut.p_cmd_we && !dut.p_cmd_full && tl_n < MAX_CMDS) begin
            u = dut.p_cmd_unit;
            tl_push[tl_n] = cyc;
            tl_unit[tl_n] = u;
            tl_op  [tl_n] = dut.p_cmd_data[7:0];
            tl_w0  [tl_n] = dut.p_cmd_data[31:0];
            tl_w1  [tl_n] = dut.p_cmd_data[63:32];
            tl_w2  [tl_n] = dut.p_cmd_data[95:64];
            if (u < 3 && dut.p_cmd_data[7:0] == tl_exec_op(u)) begin
                tl_q[u][tl_wr[u]] = tl_n;
                tl_wr[u]          = tl_wr[u] + 1;
            end
            tl_gap[tl_n] = tl_idle_acc;
            tl_idle_acc  = 0;
            tl_n         = tl_n + 1;
        end
        if (dut.mxu_start) begin
            tl_cur[0] = tl_q[0][tl_rd[0]]; tl_rd[0] = tl_rd[0] + 1;
            tl_start[tl_cur[0]] = cyc;
        end
        if (dut.vpu_start) begin
            tl_cur[1] = tl_q[1][tl_rd[1]]; tl_rd[1] = tl_rd[1] + 1;
            tl_start[tl_cur[1]] = cyc;
        end
        if (dut.dma_start) begin
            tl_cur[2] = tl_q[2][tl_rd[2]]; tl_rd[2] = tl_rd[2] + 1;
            tl_start[tl_cur[2]] = cyc;
        end
        if (dut.mxu_done && tl_cur[0] >= 0) tl_end[tl_cur[0]] = cyc;
        if (dut.vpu_done && tl_cur[1] >= 0) tl_end[tl_cur[1]] = cyc;
        if (dut.dma_done && tl_cur[2] >= 0) tl_end[tl_cur[2]] = cyc;

        // Busy attribution. `tl_cur[u]` still names the last command that unit
        // ran, so a clock is charged to it only while the unit is up -- between
        // commands every unit is idle and nothing is charged.
        if (busy) begin
            if (dut.mxu_busy && tl_cur[0] >= 0) tl_busy[tl_cur[0]] = tl_busy[tl_cur[0]] + 1;
            if (dut.vpu_busy && tl_cur[1] >= 0) tl_busy[tl_cur[1]] = tl_busy[tl_cur[1]] + 1;
            if (dut.dma_busy && tl_cur[2] >= 0) tl_busy[tl_cur[2]] = tl_busy[tl_cur[2]] + 1;
            if (!dut.mxu_busy && !dut.vpu_busy && !dut.dma_busy)
                tl_idle_acc = tl_idle_acc + 1;
        end
    end

    task automatic write_timeline();
        $fdisplay(tl_f, "# idx,unit,op,w0,w1,w2,push,start,end,busy,gap");
        // Two run lengths, and the arithmetic below wants the second: `run_clk`
        // is wall clocks from host_run to done, while the perf counters measure
        // the `busy` window and land one clock shorter. Mixing them makes the
        // reconstructed overlap come out at -1.
        $fdisplay(tl_f, "# run_clocks=%0d perf_run=%0d tail_idle=%0d",
                  run_clk, dut.u_perf.counts[0*32 +: 32], tl_idle_acc);
        for (int i = 0; i < tl_n; i++)
            $fdisplay(tl_f, "%0d,%0d,%0d,%08h,%08h,%08h,%0d,%0d,%0d,%0d,%0d",
                      i, tl_unit[i], tl_op[i], tl_w0[i], tl_w1[i], tl_w2[i],
                      tl_push[i], tl_start[i], tl_end[i], tl_busy[i], tl_gap[i]);
        $fclose(tl_f);
        $display("timeline: %0d commands -> %0s", tl_n, tl_path);
    endtask

    // ---- checks --------------------------------------------------------------
    int got;

    task automatic check_dram();
        // Every byte of DRAM, not just the result block. A byte the golden run
        // wrote must match it; every other byte must still hold what it was
        // seeded with. That subsumes the old "nothing past C" guard and is
        // kernel-agnostic — a stray write anywhere is a failure, wherever the
        // kernel happened to put its output.
        logic [7:0] want;
        for (int a = 0; a < SRAM_SZ; a++) begin
            if (dram_exp[a] !== 8'hxx)      want = dram_exp[a];
            else if (dram_in[a] !== 8'hxx)  want = dram_in[a];
            else                            want = 8'h00;
            checks++;
            if (sram_mem[a] !== want) begin
                errors++;
                if (errors <= 8)
                    $display("  FAIL DRAM[0x%05h] = 0x%02h, expected 0x%02h%s",
                             a, sram_mem[a], want,
                             (dram_exp[a] === 8'hxx) ? "  (untouched byte was written)" : "");
            end
        end
    endtask

    task automatic check_cmds();
        checks++;
        if (n_cmd_got > MAX_CMDS || n_cmd_exp >= MAX_CMDS) begin
            errors++;
            $display("  FAIL trace does not fit MAX_CMDS=%0d (got %0d, expected %0d)",
                     MAX_CMDS, n_cmd_got, n_cmd_exp);
        end
        if (n_cmd_got !== n_cmd_exp) begin
            errors++;
            $display("  FAIL command count: RTL issued %0d, expected %0d",
                     n_cmd_got, n_cmd_exp);
        end
        for (int i = 0; i < n_cmd_exp && i < n_cmd_got; i++)
            for (int w = 0; w < 5; w++) begin
                checks++;
                if (cmd_got[i*5 + w] !== cmd_exp[i*5 + w]) begin
                    errors++;
                    $display("  FAIL cmd[%0d] word %0d: RTL %08h, expected %08h",
                             i, w, cmd_got[i*5 + w], cmd_exp[i*5 + w]);
                    $display("       RTL      unit=%0d %08h %08h %08h %08h",
                             cmd_got[i*5+0], cmd_got[i*5+1], cmd_got[i*5+2],
                             cmd_got[i*5+3], cmd_got[i*5+4]);
                    $display("       expected unit=%0d %08h %08h %08h %08h",
                             cmd_exp[i*5+0], cmd_exp[i*5+1], cmd_exp[i*5+2],
                             cmd_exp[i*5+3], cmd_exp[i*5+4]);
                end
            end
        $display("command trace: %0d commands, %0d expected", n_cmd_got, n_cmd_exp);
    endtask

    // ---- +DRAMOUT=<path>: the final SRAM, for the Python driver ---------------
    //
    // The checks above are self-contained, but accel/test's Backend contract is
    // "run the kernel, hand back DRAM" — and reading it out of the simulation
    // rather than trusting the golden is what makes the RTL an independent
    // backend rather than a second opinion from the same model.
    task automatic dump_dram();
        string path;
        if ($value$plusargs("DRAMOUT=%s", path)) begin
            $writememh(path, sram_mem);
            $display("dram: %0d bytes -> %0s", SRAM_SZ, path);
        end
    endtask

    // ---- stimulus ------------------------------------------------------------
    initial begin
        $display("==== firmware matmul: %s ====", `FW_HEX);
        $display("problem: [%0dx%0d] @ [%0dx%0d], %0dx%0d tiles of %0dx%0d",
                 M, K, K, N, KTILES, NTILES, ROWS, COLS);

        load_vectors();
        seed_dram();

        host_run = 1'b0;
        boot_pc  = '0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        // FW_INIT is a $readmemh at elaboration: if the image is missing the
        // firmware RAM is all x and the CPU would run garbage until the
        // watchdog. Say so instead.
        if (dut.u_cpu.fw_mem[0] === 32'hxxxx_xxxx) begin
            $display("FW_MATMUL: %s did not load - build it first (make -C ../fw)",
                     `FW_HEX);
            $fatal(1);
        end

        // Top bit of boot_pc selects the producer: 1 = the CPU's firmware.
        @(negedge clk);
        boot_pc  = (1 << FW_AW);
        host_run = 1'b1;
        run_clk  = cyc;
        @(negedge clk);
        host_run = 1'b0;

        wait (done);
        run_clk = cyc - run_clk;
        $display("firmware signalled done after %0d clocks", run_clk);
        // Same counters the 'T' command reports. `idlec` is what CPU issue costs:
        // clocks in which no unit was busy at all.
        $display("  counters: run=%0d mxu=%0d mload=%0d vpu=%0d dma=%0d",
                 dut.u_perf.counts[0*32 +: 32], dut.u_perf.counts[1*32 +: 32],
                 dut.u_perf.counts[2*32 +: 32], dut.u_perf.counts[3*32 +: 32],
                 dut.u_perf.counts[4*32 +: 32]);
        $display("            idlec=%0d qfull=%0d ovlap=%0d",
                 dut.u_perf.counts[7*32 +: 32], dut.u_perf.counts[8*32 +: 32],
                 dut.u_perf.counts[9*32 +: 32]);
        // One machine-readable line for run_fw_sweep.sh. The command counts come
        // from the queues' own `issued`, so "how many dispatches did this shape
        // cost the CPU" is measured rather than assumed.
        $display("SWEEP m=%0d kt=%0d nt=%0d run=%0d mxu=%0d dma=%0d idlec=%0d qfull=%0d mxucmd=%0d dmacmd=%0d",
                 M, KTILES, NTILES,
                 dut.u_perf.counts[0*32 +: 32], dut.u_perf.counts[1*32 +: 32],
                 dut.u_perf.counts[4*32 +: 32], dut.u_perf.counts[7*32 +: 32],
                 dut.u_perf.counts[8*32 +: 32], dut.mxu_issued, dut.dma_issued);

        check_dram();
        check_cmds();
        if (tl_f != 0) write_timeline();
        dump_dram();

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("FW_MATMUL: ALL TESTS PASSED");
        else             $display("FW_MATMUL: %0d FAILURES", errors);
        $finish;
    end

    initial begin
        #(`WATCHDOG_NS);
        $display("FW_MATMUL: WATCHDOG - firmware never signalled done (busy=%b done=%b)",
                 busy, done);
        $fatal(1);
    end

endmodule
