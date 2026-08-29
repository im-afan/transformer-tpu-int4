`timescale 1ns/1ps
// TPU core: PicoRV32 command producer, three per-unit command queues, the
// systolic array, the vector unit, the banked scratchpad and the DMA engine.
// The DMA owns the external SRAM pins and also serves the UART host's byte
// accesses while the core is idle.

module tpu_top #(
    // ---- geometry -----------------------------------------------------------
    parameter int ROWS   = 8,     // systolic array size (must equal COLS)
    parameter int COLS   = 8,
    parameter int ADDR_W = 16,    // scratchpad byte address
    parameter int XLEN   = 32,
    parameter int M0_W   = 12,
    parameter int N_W    = 4,

    // Retained for host / testbench port-width compatibility.
    parameter int IMEM_AW = 10,
    parameter int CFG_AW  = 5,

    parameter int CMD_DEPTH = 8,
    parameter int FW_AW     = 12,
    parameter     FW_INIT   = "",

    parameter int MEM_ADDR_W = 19,
    parameter int MEM_DATA_W = 8,

    parameter int UART_CPB        = 104,
    parameter int UART_RX_TIMEOUT = 0,

    parameter     MEM_STYLE       = "BRAM",
    parameter     SPAD_INIT       = "",
    parameter int SPAD_BANK_WORDS = 1024
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                 host_run,
    input  logic [FW_AW:0]       boot_pc,
    output logic                 busy,
    output logic                 done,
    output logic [IMEM_AW-1:0]   pc_dbg,

    input  logic                 imem_we,
    input  logic [IMEM_AW-1:0]   imem_waddr,
    input  logic [31:0]          imem_wdata,

    input  logic                 cfg_we,
    input  logic [CFG_AW-1:0]    cfg_waddr,
    input  logic [XLEN-1:0]      cfg_wdata,

    output logic [MEM_ADDR_W-1:0] sram_addr,
    inout  wire  [MEM_DATA_W-1:0] sram_data,
    output logic                  sram_we,
    output logic                  sram_ce,
    output logic                  sram_oen,

    input  logic                  uart_rx,
    output logic                  uart_tx
);

    localparam int N          = ROWS;
    localparam int WORD_BITS  = N * 4;
    localparam int WORD_BYTES = N / 2;
    localparam int S_BYTES    = XLEN / 8;

    initial if (ROWS != COLS)
        $fatal(1, "tpu_top: the array is square — ROWS (%0d) != COLS (%0d)", ROWS, COLS);

    // =========================================================================
    // Interconnect.
    // =========================================================================
    logic                mxu_start, mxu_transpose, mxu_accumulate;
    logic [15:0]         mxu_len;
    logic [ADDR_W-1:0]   mxu_a_base, mxu_a_stride;
    logic [ADDR_W-1:0]   mxu_b_base, mxu_b_stride;
    logic [ADDR_W-1:0]   mxu_c_base, mxu_c_stride;
    logic [M0_W+N_W-1:0] mxu_rq_word;
    logic                mxu_busy, mxu_done;

    logic                vpu_start;
    logic [4:0]          vpu_op;
    logic [ADDR_W-1:0]   vpu_src0, vpu_src1, vpu_dst;
    logic [M0_W+N_W-1:0] vpu_rq_word;
    logic [9:0]          vpu_vlen;
    logic                vpu_busy, vpu_done;

    localparam logic [1:0] U_MXU = 2'd0, U_VPU = 2'd1, U_DMA = 2'd2, U_LINK = 2'd3;

    logic         cpu_cmd_we, cpu_cmd_full;
    logic [1:0]   cpu_cmd_unit;
    logic [127:0] cpu_cmd_data;

    logic         p_cmd_we, p_cmd_full;
    logic [1:0]   p_cmd_unit;
    logic [127:0] p_cmd_data;

    logic        mxu_cmd_we, vpu_cmd_we, dma_cmd_we;
    logic        mxu_cmd_full, vpu_cmd_full, dma_cmd_full;
    logic [31:0] mxu_issued, mxu_retired, vpu_issued, vpu_retired,
                 dma_issued, dma_retired;
    logic [15:0] mxu_level, vpu_level, dma_level;
    logic        mxu_idle, vpu_idle, dma_idle;
    logic [3:0]  unit_idle;

    // scratchpad ports
    logic                  A_re, A_gnt;
    logic [ADDR_W-1:0]     A_addr;
    logic [WORD_BITS-1:0]  A_rdata;

    logic                  B_re, B_gnt;
    logic [ADDR_W-1:0]     B_addr;
    logic [WORD_BITS-1:0]  B_rdata;

    logic                  C_en, C_we, C_gnt;
    logic [ADDR_W-1:0]     C_addr;
    logic [WORD_BITS-1:0]  C_wdata, C_rdata;

    logic                  V_re, V_we, V_rgnt, V_wgnt;
    logic [ADDR_W-1:0]     V_raddr, V_waddr;
    logic [WORD_BITS-1:0]  V_rdata, V_wdata;
    logic [WORD_BYTES-1:0] V_wstrb;

    logic                  spad_dma_re, spad_dma_we, spad_dma_rgnt, spad_dma_wgnt;
    logic [ADDR_W-1:0]     spad_dma_raddr, spad_dma_waddr;
    logic [7:0]            spad_dma_rdata, spad_dma_wdata;

    logic                s_re, s_we, s_rgnt, s_wgnt;
    logic [ADDR_W-1:0]   s_addr;
    logic [XLEN-1:0]     s_wdata, s_rdata;

    logic                cpu_s_re, cpu_s_we, cpu_s_rgnt, cpu_s_wgnt;
    logic [ADDR_W-1:0]   cpu_s_addr;
    logic [XLEN-1:0]     cpu_s_wdata;

    // DMA dispatch
    logic                  dma_start, dma_op;
    logic [15:0]           dma_len, dma_rows, dma_dram_stride, dma_spad_stride;
    logic [MEM_ADDR_W-1:0] dma_dram_base;
    logic [ADDR_W-1:0]     dma_spad_base;
    logic                  dma_busy, dma_done;

    // UART host byte port into the DMA
    logic                  uart_mem_start, uart_mem_we;
    logic [MEM_ADDR_W-1:0] uart_mem_addr;
    logic [7:0]            uart_mem_din, host_dout;
    logic                  host_busy, host_done;

    logic [7:0]            uart_rx_data;
    logic                  uart_rx_valid;
    logic                  uart_tx_start, uart_tx_busy;
    logic [7:0]            uart_tx_data;

    logic                  uart_imem_we;
    logic [FW_AW:0]        uart_imem_waddr;
    logic [31:0]           uart_imem_wdata;
    logic                  uart_run_start;
    logic [FW_AW:0]        uart_run_pc;

    localparam int NPERF = 10;
    logic [NPERF-1:0]    perf_ev;
    logic [NPERF*32-1:0] perf_counts, perf_wire;

    // =========================================================================
    // Host program load + run.
    // =========================================================================
    localparam int HOST_AW = FW_AW + 1;

    logic             fw_we;
    logic [FW_AW-1:0] fw_waddr, fw_waddr_host, fw_waddr_uart;
    logic [31:0]      fw_wdata;
    logic             cpu_run, cpu_busy, cpu_done, cpu_trap;

    assign fw_waddr_host = imem_waddr;
    assign fw_waddr_uart = uart_imem_waddr;

    assign fw_we    = imem_we | uart_imem_we;
    assign fw_waddr = uart_imem_we ? fw_waddr_uart : fw_waddr_host;
    assign fw_wdata = uart_imem_we ? uart_imem_wdata : imem_wdata;

    // `cpu_started` is what makes a second 'G' work with no reset in between:
    // cpu_done is level-held from the previous run and only clears one clock
    // after cpu_run rises.
    logic cpu_started;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cpu_run     <= 1'b0;
            cpu_started <= 1'b0;
        end else if (uart_run_start || host_run) begin
            cpu_run     <= 1'b1;
            cpu_started <= 1'b0;
        end else begin
            if (cpu_busy)                           cpu_started <= 1'b1;
            if (cpu_run && cpu_started && cpu_done) cpu_run <= 1'b0;
        end
    end

    assign busy   = cpu_busy;
    assign done   = cpu_done;
    assign pc_dbg = '0;

    // =========================================================================
    // Command plane.
    // =========================================================================
    assign p_cmd_we   = cpu_cmd_we;
    assign p_cmd_unit = cpu_cmd_unit;
    assign p_cmd_data = cpu_cmd_data;

    always_comb begin
        unique case (p_cmd_unit)
            U_MXU:   p_cmd_full = mxu_cmd_full;
            U_VPU:   p_cmd_full = vpu_cmd_full;
            default: p_cmd_full = dma_cmd_full;
        endcase
    end

    assign cpu_cmd_full = p_cmd_full;

    assign mxu_cmd_we = p_cmd_we && (p_cmd_unit == U_MXU);
    assign vpu_cmd_we = p_cmd_we && (p_cmd_unit == U_VPU);
    assign dma_cmd_we = p_cmd_we && (p_cmd_unit == U_DMA);

    assign unit_idle = {1'b1, dma_idle, vpu_idle, mxu_idle};

    cmd_mxu #(
        .ADDR_W (ADDR_W), .M0_W (M0_W), .N_W (N_W), .DEPTH (CMD_DEPTH)
    ) u_cmd_mxu (
        .clk (clk), .rst_n (rst_n),
        .cmd_we (mxu_cmd_we), .cmd_wdata (p_cmd_data), .cmd_full (mxu_cmd_full),
        .mxu_start      (mxu_start),
        .mxu_transpose  (mxu_transpose),
        .mxu_accumulate (mxu_accumulate),
        .mxu_len        (mxu_len),
        .mxu_a_base     (mxu_a_base),
        .mxu_a_stride   (mxu_a_stride),
        .mxu_b_base     (mxu_b_base),
        .mxu_b_stride   (mxu_b_stride),
        .mxu_c_base     (mxu_c_base),
        .mxu_c_stride   (mxu_c_stride),
        .mxu_rq_word    (mxu_rq_word),
        .mxu_done       (mxu_done),
        .issued (mxu_issued), .retired (mxu_retired), .level (mxu_level),
        .idle   (mxu_idle)
    );

    cmd_vpu #(
        .ADDR_W (ADDR_W), .M0_W (M0_W), .N_W (N_W), .DEPTH (CMD_DEPTH)
    ) u_cmd_vpu (
        .clk (clk), .rst_n (rst_n),
        .cmd_we (vpu_cmd_we), .cmd_wdata (p_cmd_data), .cmd_full (vpu_cmd_full),
        .vpu_start   (vpu_start),
        .vpu_op      (vpu_op),
        .vpu_src0    (vpu_src0),
        .vpu_src1    (vpu_src1),
        .vpu_rq_word (vpu_rq_word),
        .vpu_dst     (vpu_dst),
        .vpu_vlen    (vpu_vlen),
        .vpu_done    (vpu_done),
        .issued (vpu_issued), .retired (vpu_retired), .level (vpu_level),
        .idle   (vpu_idle)
    );

    cmd_dma #(
        .ADDR_W (ADDR_W), .MEM_ADDR_W (MEM_ADDR_W), .DEPTH (CMD_DEPTH)
    ) u_cmd_dma (
        .clk (clk), .rst_n (rst_n),
        .cmd_we (dma_cmd_we), .cmd_wdata (p_cmd_data), .cmd_full (dma_cmd_full),
        .dma_start       (dma_start),
        .dma_op          (dma_op),
        .dma_len         (dma_len),
        .dma_rows        (dma_rows),
        .dma_dram_stride (dma_dram_stride),
        .dma_spad_stride (dma_spad_stride),
        .dma_dram_base   (dma_dram_base),
        .dma_spad_base   (dma_spad_base),
        .dma_done        (dma_done),
        .issued (dma_issued), .retired (dma_retired), .level (dma_level),
        .idle   (dma_idle)
    );

    // =========================================================================
    // PicoRV32 subsystem — the command producer.
    // =========================================================================
    cpu_subsys #(
        .ADDR_W (ADDR_W), .MEM_ADDR_W (MEM_ADDR_W), .XLEN (XLEN),
        .FW_AW  (FW_AW),  .FW_INIT (FW_INIT)
    ) u_cpu (
        .clk (clk), .rst_n (rst_n),
        .cpu_run (cpu_run), .cpu_busy (cpu_busy),
        .cpu_done (cpu_done), .cpu_trap (cpu_trap),
        .fw_we (fw_we), .fw_waddr (fw_waddr), .fw_wdata (fw_wdata),
        .cmd_we (cpu_cmd_we), .cmd_unit (cpu_cmd_unit),
        .cmd_data (cpu_cmd_data), .cmd_full (cpu_cmd_full),
        .mxu_issued (mxu_issued), .mxu_retired (mxu_retired),
        .vpu_issued (vpu_issued), .vpu_retired (vpu_retired),
        .dma_issued (dma_issued), .dma_retired (dma_retired),
        .mxu_level  (mxu_level), .vpu_level (vpu_level), .dma_level (dma_level),
        .unit_idle  (unit_idle),
        .s_re (cpu_s_re), .s_we (cpu_s_we), .s_addr (cpu_s_addr),
        .s_wdata (cpu_s_wdata), .s_rdata (s_rdata),
        .s_rgnt (cpu_s_rgnt), .s_wgnt (cpu_s_wgnt)
    );

    assign s_re    = cpu_s_re;
    assign s_we    = cpu_s_we;
    assign s_addr  = cpu_s_addr;
    assign s_wdata = cpu_s_wdata;

    assign cpu_s_rgnt = s_rgnt;
    assign cpu_s_wgnt = s_wgnt;

    // =========================================================================
    // MXU.
    // =========================================================================
    mxu #(
        .N (N), .ADDR_W (ADDR_W), .M0_W (M0_W), .N_W (N_W)
    ) u_mxu (
        .clk (clk), .rst_n (rst_n),
        .start      (mxu_start),
        .transpose  (mxu_transpose),
        .accumulate (mxu_accumulate),
        .len        (mxu_len),
        .a_base     (mxu_a_base),
        .a_stride   (mxu_a_stride),
        .b_base     (mxu_b_base),
        .b_stride   (mxu_b_stride),
        .c_base     (mxu_c_base),
        .c_stride   (mxu_c_stride),
        .rq_word    (mxu_rq_word),
        .busy       (mxu_busy),
        .done       (mxu_done),
        .A_re (A_re), .A_addr (A_addr), .A_rdata (A_rdata), .A_gnt (A_gnt),
        .B_re (B_re), .B_addr (B_addr), .B_rdata (B_rdata), .B_gnt (B_gnt),
        .C_en (C_en), .C_we (C_we), .C_addr (C_addr),
        .C_wdata (C_wdata), .C_rdata (C_rdata), .C_gnt (C_gnt)
    );

    // =========================================================================
    // VPU. Its port is one scratchpad word wide, like everything else.
    // =========================================================================
    vpu #(
        .SCRATCHPAD_W (WORD_BYTES),
        .ADDR_W       (ADDR_W),
        .M0_W         (M0_W),
        .N_W          (N_W)
    ) u_vpu (
        .clk (clk), .rst_n (rst_n),
        .vpu_start   (vpu_start),
        .vpu_op      (vpu_op),
        .vpu_src0    (vpu_src0),
        .vpu_src1    (vpu_src1),
        .vpu_rq_word (vpu_rq_word),
        .vpu_dst     (vpu_dst),
        .vpu_vlen    (vpu_vlen),
        .vpu_busy    (vpu_busy),
        .vpu_done    (vpu_done),
        .V_re (V_re), .V_raddr (V_raddr), .V_rdata (V_rdata),
        .V_we (V_we), .V_waddr (V_waddr), .V_wdata (V_wdata), .V_wstrb (V_wstrb),
        .V_rgnt (V_rgnt), .V_wgnt (V_wgnt)
    );

    // =========================================================================
    // Scratchpad.
    // =========================================================================
    scratchpad #(
        .MEM_STYLE  (MEM_STYLE),
        .N          (N),
        .ADDR_W     (ADDR_W),
        .BANK_WORDS (SPAD_BANK_WORDS),
        .S_BYTES    (S_BYTES),
        .INIT_FILE  (SPAD_INIT)
    ) u_scratchpad (
        .clk (clk), .rst_n (rst_n),
        .A_re (A_re), .A_addr (A_addr), .A_rdata (A_rdata), .A_gnt (A_gnt),
        .B_re (B_re), .B_addr (B_addr), .B_rdata (B_rdata), .B_gnt (B_gnt),
        .C_en (C_en), .C_we (C_we), .C_addr (C_addr),
        .C_wdata (C_wdata), .C_rdata (C_rdata), .C_gnt (C_gnt),
        .V_re (V_re), .V_raddr (V_raddr), .V_rdata (V_rdata), .V_rgnt (V_rgnt),
        .V_we (V_we), .V_waddr (V_waddr), .V_wdata (V_wdata),
        .V_wstrb (V_wstrb), .V_wgnt (V_wgnt),
        .dma_re (spad_dma_re), .dma_raddr (spad_dma_raddr),
        .dma_rdata (spad_dma_rdata), .dma_rgnt (spad_dma_rgnt),
        .dma_we (spad_dma_we), .dma_waddr (spad_dma_waddr),
        .dma_wdata (spad_dma_wdata), .dma_wgnt (spad_dma_wgnt),
        .s_re (s_re), .s_we (s_we), .s_addr (s_addr),
        .s_wdata (s_wdata), .s_rdata (s_rdata),
        .s_rgnt (s_rgnt), .s_wgnt (s_wgnt)
    );

    // =========================================================================
    // DMA — owns the SRAM pins, and the UART host's byte port while idle.
    // =========================================================================
    dma #(
        .MEM_ADDR_W (MEM_ADDR_W),
        .ADDR_W     (ADDR_W)
    ) u_dma (
        .clk (clk), .rst_n (rst_n),
        .sram_addr (sram_addr), .sram_data (sram_data),
        .sram_we (sram_we), .sram_ce (sram_ce), .sram_oen (sram_oen),
        .dma_start       (dma_start),
        .dma_op          (dma_op),
        .dma_len         (dma_len),
        .dma_rows        (dma_rows),
        .dma_dram_stride (dma_dram_stride),
        .dma_spad_stride (dma_spad_stride),
        .dma_dram_base   (dma_dram_base),
        .dma_spad_base   (dma_spad_base),
        .dma_busy        (dma_busy),
        .dma_done        (dma_done),
        .spad_re (spad_dma_re), .spad_raddr (spad_dma_raddr),
        .spad_rdata (spad_dma_rdata), .spad_rgnt (spad_dma_rgnt),
        .spad_we (spad_dma_we), .spad_waddr (spad_dma_waddr),
        .spad_wdata (spad_dma_wdata), .spad_wgnt (spad_dma_wgnt),
        .host_start (uart_mem_start), .host_we (uart_mem_we),
        .host_addr (uart_mem_addr), .host_din (uart_mem_din),
        .host_dout (host_dout), .host_busy (host_busy), .host_done (host_done)
    );

    // =========================================================================
    // Performance counters — ten event bits integrated over one run, read over
    // UART with 'T'. Indices 2, 5 and 6 are retired and tied low so the reply
    // does not renumber.
    //
    //   0 run   1 mxu   2 (mload, retired)   3 vpu   4 dma
    //   5 (swait, retired)   6 (vmm, retired)   7 idlec   8 qfull   9 ovlap
    // =========================================================================
    localparam int PERF_RUN = 0, PERF_MXU  = 1, PERF_MLOAD = 2,
                   PERF_VPU = 3, PERF_DMA  = 4, PERF_SWAIT = 5,
                   PERF_VMM = 6, PERF_IDLEC = 7, PERF_QFULL = 8,
                   PERF_OVLAP = 9;

    wire [1:0] n_busy = 2'(mxu_busy) + 2'(vpu_busy) + 2'(dma_busy);

    always_comb begin
        perf_ev             = '0;
        perf_ev[PERF_RUN]   = busy;
        perf_ev[PERF_MXU]   = mxu_busy;
        perf_ev[PERF_MLOAD] = 1'b0;   // was the weight-load phase
        perf_ev[PERF_VPU]   = vpu_busy;
        perf_ev[PERF_DMA]   = dma_busy;
        perf_ev[PERF_SWAIT] = 1'b0;   // was the scalar unit's issue-and-wait stall
        perf_ev[PERF_VMM]   = 1'b0;   // was the VPU's vecmatmul macro op
        perf_ev[PERF_IDLEC] = (n_busy == 2'd0);
        perf_ev[PERF_QFULL] = p_cmd_we & p_cmd_full;
        perf_ev[PERF_OVLAP] = (n_busy >= 2'd2);
    end

    perf_counters #(
        .N (NPERF), .W (32)
    ) u_perf (
        .clk (clk), .rst_n (rst_n), .run (busy),
        .ev (perf_ev), .counts (perf_counts)
    );

    // uart_interface shifts the 'T' reply out of the high end, so reverse the
    // word order to put counter 0 on the wire first.
    always_comb begin
        for (int i = 0; i < NPERF; i++)
            perf_wire[(NPERF-1-i)*32 +: 32] = perf_counts[i*32 +: 32];
    end

    // =========================================================================
    // UART host link.
    // =========================================================================
    uart_receiver #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_rx (
        .clk (clk), .rst_n (rst_n),
        .uart_rx (uart_rx), .data (uart_rx_data), .valid (uart_rx_valid)
    );

    uart_transmitter #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_tx (
        .clk (clk), .rst_n (rst_n),
        .start (uart_tx_start), .data (uart_tx_data),
        .uart_tx (uart_tx), .busy (uart_tx_busy)
    );

    uart_interface #(
        .ADDR_W      (MEM_ADDR_W),
        .LENGTH_W    (16),
        .IMEM_AW     (HOST_AW),
        .RX_TIMEOUT  (UART_RX_TIMEOUT),
        .TIMER_WORDS (NPERF)
    ) u_uart (
        .clk (clk), .rst_n (rst_n),
        .core_busy   (busy),
        .cycle_count (perf_wire),
        .data_in           (uart_rx_data),
        .receiver_valid    (uart_rx_valid),
        .transmitter_start (uart_tx_start),
        .data_out          (uart_tx_data),
        .transmitter_busy  (uart_tx_busy),
        // the DMA's host byte port
        .sram_start (uart_mem_start),
        .sram_we    (uart_mem_we),
        .sram_addr  (uart_mem_addr),
        .sram_din   (uart_mem_din),
        .sram_dout  (host_dout),
        .sram_busy  (host_busy),
        .sram_done  (host_done),
        .imem_we    (uart_imem_we),
        .imem_waddr (uart_imem_waddr),
        .imem_wdata (uart_imem_wdata),
        .run_start (uart_run_start),
        .run_pc    (uart_run_pc),
        .host_busy  (),
        .rx_overrun ()
    );

endmodule
