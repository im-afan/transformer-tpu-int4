// UART block-loopback self-test core: receives BLOCK_LEN bytes, sends them
// back, repeats. See docs/uart_selftest.md.

module uart_echo #(
    parameter int CLK_PER_BIT = 104,  // 12 MHz / 115200 baud
    parameter int BLOCK_LEN   = 64,   // bytes buffered per exchange
    parameter int ACT_W       = 19,   // activity LED stretch: 2**19 / 12 MHz ~= 44 ms
    parameter int HB_W        = 23    // heartbeat: 12 MHz / 2**23 ~= 1.4 Hz
) (
    input  logic clk,
    input  logic rst_n,

    input  logic uart_rx,
    output logic uart_tx,

    // Status, for the board wrapper's LEDs.
    output logic blink_slow,   // free-running: the clock and the bitstream are alive
    output logic blink_fast,   // free-running, ~8x faster: used to flag `overrun`
    output logic activity,     // stretched high for ACT_W clocks per received byte
    output logic overrun       // sticky: a byte arrived while the reply was going out
);

    // Index width for 0..BLOCK_LEN-1. BLOCK_LEN need not be a power of two; the
    // wrap is an explicit compare against BLOCK_LEN-1, not a counter rollover.
    localparam int CNT_W = (BLOCK_LEN <= 1) ? 1 : $clog2(BLOCK_LEN);

    // =========================================================================
    // Receiver / transmitter — the production blocks, untouched.
    // =========================================================================
    logic [7:0] rx_data;
    logic       rx_valid;

    uart_receiver #(
        .CLK_PER_BIT (CLK_PER_BIT)
    ) u_rx (
        .clk     (clk),
        .rst_n   (rst_n),
        .uart_rx (uart_rx),
        .data    (rx_data),
        .valid   (rx_valid)
    );

    logic [7:0] tx_data;
    logic       tx_start, tx_busy;

    uart_transmitter #(
        .CLK_PER_BIT (CLK_PER_BIT)
    ) u_tx (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (tx_start),
        .data    (tx_data),
        .uart_tx (uart_tx),
        .busy    (tx_busy)
    );

    // rx_valid is a level (STOP through half the next start bit), so this is
    // the rising edge — one pulse per byte, matching uart_interface's rx_byte.
    logic rx_valid_prev;
    wire  rx_byte = rx_valid & ~rx_valid_prev;

    // RECV/SEND sequencer; see docs/uart_selftest.md "Design notes".
    logic [7:0]       block_mem [0:BLOCK_LEN-1];
    logic [CNT_W-1:0] cnt;

    // Encoded as localparams rather than an enum, matching uart_receiver.sv and
    // uart_interface.sv.
    localparam RECV = 1'b0,   // filling block_mem, silent
               SEND = 1'b1;   // draining block_mem, deaf

    logic state;

    // `cnt` is unsigned and zero-extends to the integer compare, so this is a
    // plain equality against the last index in either phase.
    wire last = (cnt == BLOCK_LEN - 1);

    logic [ACT_W-1:0] act_cnt;
    logic [HB_W-1:0]  hb_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_valid_prev <= 1'b0;
            state         <= RECV;
            cnt           <= '0;
            tx_start      <= 1'b0;
            tx_data       <= 8'b0;
            overrun       <= 1'b0;
            act_cnt       <= '0;
            hb_cnt        <= '0;
        end else begin
            rx_valid_prev <= rx_valid;
            hb_cnt        <= hb_cnt + 1'b1;

            // `tx_start` is a one-cycle strobe; SEND re-raises it below.
            tx_start <= 1'b0;

            case (state)
                RECV: begin
                    if (rx_byte) begin
                        block_mem[cnt] <= rx_data;
                        if (last) begin
                            cnt   <= '0;
                            state <= SEND;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                    end
                end

                SEND: begin
                    // Nothing is listening this phase; flag it rather than drop it silently.
                    if (rx_byte) overrun <= 1'b1;

                    // !tx_start covers the cycle after the strobe, before `busy` asserts.
                    if (!tx_start && !tx_busy) begin
                        tx_data  <= block_mem[cnt];
                        tx_start <= 1'b1;
                        if (last) begin
                            cnt   <= '0;
                            state <= RECV;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                    end
                end
            endcase

            // ---- activity LED, independent of phase -------------------------
            if (rx_byte) act_cnt <= '1;
            else if (act_cnt != 0) act_cnt <= act_cnt - 1'b1;
        end
    end

    assign blink_slow = hb_cnt[HB_W-1];
    assign blink_fast = hb_cnt[HB_W-4];
    assign activity   = (act_cnt != 0);

endmodule
