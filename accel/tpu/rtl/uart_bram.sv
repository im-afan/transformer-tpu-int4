// UART command link + on-chip block RAM. uart_memory.sv with the external
// memory swapped for bram_controller. See docs/uart_selftest.md.

module uart_bram #(
    // ---- UART host link ------------------------------------------------------
    parameter int UART_CPB        = 104,  // 12 MHz / 115200 baud
    parameter int UART_RX_TIMEOUT = 0,    // inter-byte abort (clocks; 0 = off)

    // ---- Memory --------------------------------------------------------------
    //   MEM_ADDR_W is the protocol address width and must stay 19 to match
    //   boards/cmod_a7 — it is what uart_interface range-checks against, and a
    //   rig that rejects different frames than the real image is not a control.
    //   BRAM_AW is how much of that space is actually built.
    parameter int MEM_ADDR_W      = 19,
    parameter int MEM_DATA_W      = 8,
    parameter int BRAM_AW         = 16,   // 2**16 = 64 KiB = 16 RAMB36 of 50
    parameter int MEM_CPA         = 0,    // = boards/cmod_a7_mem's SRAM_CPA

    // ---- Instruction memory address width -----------------------------------
    //   No IMEM exists in this image; this only has to match what the host's
    //   'I' frames would be range-checked against.
    parameter int IMEM_AW         = 10,

    // ---- Status LED timing ---------------------------------------------------
    parameter int ACT_W           = 19,   // activity stretch: 2**19 / 12 MHz ~= 44 ms
    parameter int HB_W            = 23    // heartbeat: 12 MHz / 2**23 ~= 1.4 Hz
) (
    input  logic clk,
    input  logic rst_n,

    // ---- USB-serial link -----------------------------------------------------
    input  logic uart_rx,
    output logic uart_tx,

    // ---- Status, for the board wrapper's LEDs -------------------------------
    output logic blink_slow,   // free-running: the clock and the bitstream are alive
    output logic blink_fast,   // free-running, ~8x faster: used to flag the two below
    output logic activity,     // stretched high for ACT_W clocks per received byte
    output logic collision,    // sticky: see below
    output logic aliased       // sticky: an access landed outside the built window
);

    // =========================================================================
    // Memory controller user side. One requester, so no mux — same as
    // uart_memory: a mux with one input is a thing that can be wrong.
    // =========================================================================
    logic                   mem_start, mem_we, mem_busy, mem_done;
    logic [MEM_ADDR_W-1:0]  mem_addr;
    logic [MEM_DATA_W-1:0]  mem_din, mem_dout;

    // ---- UART host interface -------------------------------------------------
    logic [7:0]             uart_rx_data;
    logic                   uart_rx_valid;
    logic                   uart_tx_start, uart_tx_busy;
    logic [7:0]             uart_tx_data;
    logic                   uart_host_busy;

    // IMEM write port and run trigger. Decoded and driven by uart_interface, and
    // then dropped: there is no scalar unit in this image. Declared rather than
    // left as unnamed open ports so the dead end is explicit in the netlist.
    logic                   uart_imem_we;
    logic [IMEM_AW-1:0]     uart_imem_waddr;
    logic [31:0]            uart_imem_wdata;
    logic                   uart_run_start;
    logic [IMEM_AW-1:0]     uart_run_pc;
    logic                   uart_rx_overrun;

    // =========================================================================
    // Block RAM — the only line that differs from uart_memory.sv.
    // =========================================================================
    bram_controller #(
        .CLOCKS_PER_ACCESS (MEM_CPA),
        .ADDR_W            (MEM_ADDR_W),
        .DATA_W            (MEM_DATA_W),
        .DEPTH_W           (BRAM_AW)
    ) u_bram (
        .clk   (clk),
        .rst_n (rst_n),

        // user side ← UART host (sole owner), one byte per transaction: the
        // range interface at len = 1 (see uart_memory.sv for the same wiring).
        .start  (mem_start),
        .we     (mem_we),
        .addr   (mem_addr),
        .len    (16'd1),
        .stride (16'd0),

        .din       (mem_din),
        .din_valid (1'b1),
        .din_ready (),

        .dout       (mem_dout),
        .dout_valid (),
        .dout_ready (1'b1),   // host path takes every byte

        .busy (mem_busy),
        .done (mem_done),

        .aliased (aliased)
    );

    // =========================================================================
    // UART host link — the production blocks, untouched.
    // =========================================================================
    uart_receiver #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_rx (
        .clk     (clk),
        .rst_n   (rst_n),
        .uart_rx (uart_rx),
        .data    (uart_rx_data),
        .valid   (uart_rx_valid)
    );

    uart_transmitter #(
        .CLK_PER_BIT (UART_CPB)
    ) u_uart_tx (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (uart_tx_start),
        .data    (uart_tx_data),
        .uart_tx (uart_tx),
        .busy    (uart_tx_busy)
    );

    uart_interface #(
        .ADDR_W     (MEM_ADDR_W),
        .LENGTH_W   (16),
        .IMEM_AW    (IMEM_AW),
        .RX_TIMEOUT (UART_RX_TIMEOUT)
    ) u_uart (
        .clk   (clk),
        .rst_n (rst_n),

        // No core to arbitrate against: nothing is ever NAK'd for being busy.
        .core_busy (1'b0),

        // No scalar unit, so no run to time: 'T' is still decoded and still
        // answers, and it answers 0 forever.
        .cycle_count (32'd0),

        // receiver / transmitter
        .data_in           (uart_rx_data),
        .receiver_valid    (uart_rx_valid),
        .transmitter_start (uart_tx_start),
        .data_out          (uart_tx_data),
        .transmitter_busy  (uart_tx_busy),

        // memory controller user side (sole owner here). The port names are
        // uart_interface's and still say `sram_` — that block is unmodified on
        // purpose, and what it is talking to is not its business.
        .sram_start (mem_start),
        .sram_we    (mem_we),
        .sram_addr  (mem_addr),
        .sram_din   (mem_din),
        .sram_dout  (mem_dout),
        .sram_busy  (mem_busy),
        .sram_done  (mem_done),

        // instruction-memory write port → nowhere (no scalar unit)
        .imem_we    (uart_imem_we),
        .imem_waddr (uart_imem_waddr),
        .imem_wdata (uart_imem_wdata),

        // run trigger → nowhere (no scalar unit)
        .run_start  (uart_run_start),
        .run_pc     (uart_run_pc),

        .host_busy  (uart_host_busy),
        .rx_overrun (uart_rx_overrun)
    );

    // Instrumentation, identical to uart_memory.sv. See docs/uart_selftest.md
    // for `collision`'s two trigger conditions.
    logic uart_rx_valid_prev;
    wire  rx_byte = uart_rx_valid & ~uart_rx_valid_prev;   // rising edge of receiver valid

    // The transmitter asserts `busy` the cycle *after* it accepts `start`, so
    // the strobe has to be OR'd in or the first cycle of every frame would look
    // idle.
    wire  tx_active = uart_tx_busy | uart_tx_start;

    logic [ACT_W-1:0] act_cnt;
    logic [HB_W-1:0]  hb_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_rx_valid_prev <= 1'b0;
            act_cnt            <= '0;
            hb_cnt             <= '0;
            collision          <= 1'b0;
        end else begin
            uart_rx_valid_prev <= uart_rx_valid;
            hb_cnt             <= hb_cnt + 1'b1;

            if (uart_rx_overrun) collision <= 1'b1;   // sticky until reset

            if (rx_byte) begin
                act_cnt <= '1;
                if (tx_active) collision <= 1'b1;   // sticky until reset
            end else if (act_cnt != 0) begin
                act_cnt <= act_cnt - 1'b1;
            end
        end
    end

    assign blink_slow = hb_cnt[HB_W-1];
    assign blink_fast = hb_cnt[HB_W-4];
    assign activity   = (act_cnt != 0);

endmodule
