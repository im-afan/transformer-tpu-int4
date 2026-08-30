`timescale 1ns/1ps
// Generic command FIFO in front of each accelerator (cmd_mxu.sv, cmd_vpu.sv,
// cmd_dma.sv). See docs/macro_ops.md "Queues and synchronization".

module cmd_queue #(
    parameter int WIDTH = 128,   // command width in bits
    parameter int DEPTH = 8      // entries; must be a power of two
) (
    input  logic             clk,
    input  logic             rst_n,

    // ---- producer side ------------------------------------------------------
    input  logic             wr_en,     // ignored when `full`
    input  logic [WIDTH-1:0] wr_data,
    output logic             full,

    // ---- consumer side ------------------------------------------------------
    output logic             empty,
    output logic [WIDTH-1:0] head,      // valid whenever !empty
    input  logic             pop,       // retire `head` (ignored when empty)

    // ---- observation --------------------------------------------------------
    output logic [15:0]      count      // entries currently held
);

    localparam int PTR_W = $clog2(DEPTH);

    logic [WIDTH-1:0]  mem [0:DEPTH-1];
    logic [PTR_W-1:0]  wptr, rptr;
    logic [PTR_W:0]    level;           // one bit wider: 0..DEPTH inclusive

    assign empty = (level == '0);
    assign full  = (level == (PTR_W+1)'(DEPTH));
    assign head  = mem[rptr];
    assign count = 16'(level);

    wire do_wr = wr_en && !full;
    wire do_rd = pop   && !empty;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr  <= '0;
            rptr  <= '0;
            level <= '0;
        end else begin
            if (do_wr) begin
                mem[wptr] <= wr_data;
                wptr      <= wptr + PTR_W'(1);
            end
            if (do_rd) rptr <= rptr + PTR_W'(1);

            case ({do_wr, do_rd})
                2'b10:   level <= level + (PTR_W+1)'(1);
                2'b01:   level <= level - (PTR_W+1)'(1);
                default: ;                       // both or neither: unchanged
            endcase
        end
    end

// synthesis translate_off
`ifndef SYNTHESIS
    // A dropped write means the producer pushed through `full` instead of stalling on it.
    always @(posedge clk) begin
        if (rst_n && wr_en && full)
            $display("[%0t] cmd_queue: WRITE DROPPED, queue full (depth %0d)", $time, DEPTH);
    end
`endif
// synthesis translate_on

endmodule
