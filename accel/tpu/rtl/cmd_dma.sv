`timescale 1ns/1ps
// DMA command front end: 128-bit queue, decode, issue-and-wait. A transfer is
// self-contained — direction, geometry and both base addresses ride in the one
// command. See docs/dma.md.

module cmd_dma #(
    parameter int ADDR_W     = 16,
    parameter int MEM_ADDR_W = 19,
    parameter int DEPTH      = 8
) (
    input  logic clk,
    input  logic rst_n,

    input  logic         cmd_we,
    input  logic [127:0] cmd_wdata,
    output logic         cmd_full,

    output logic                  dma_start,
    output logic                  dma_op,
    output logic [15:0]           dma_len,
    output logic [15:0]           dma_rows,
    output logic [15:0]           dma_dram_stride,
    output logic [15:0]           dma_spad_stride,
    output logic [MEM_ADDR_W-1:0] dma_dram_base,
    output logic [ADDR_W-1:0]     dma_spad_base,
    input  logic                  dma_done,

    output logic [31:0] issued,
    output logic [31:0] retired,
    output logic [15:0] level,
    output logic        idle
);

    localparam logic [7:0] DMA_MOVE = 8'h01;

    logic [127:0] head;
    logic         empty, pop;

    cmd_queue #(.WIDTH(128), .DEPTH(DEPTH)) u_q (
        .clk (clk), .rst_n (rst_n),
        .wr_en (cmd_we), .wr_data (cmd_wdata), .full (cmd_full),
        .empty (empty), .head (head), .pop (pop),
        .count (level)
    );

    wire [7:0]  c_op = head[7:0];
    wire [31:0] w0   = head[31:0];
    wire [31:0] w1   = head[63:32];
    wire [31:0] w2   = head[95:64];
    wire [31:0] w3   = head[127:96];

    wire is_move = (c_op == DMA_MOVE);

    assign dma_op          = w0[8];
    assign dma_spad_base   = w0[31:16];
    assign dma_dram_base   = w1[MEM_ADDR_W-1:0];
    assign dma_len         = w2[15:0];
    assign dma_rows        = w2[31:16];
    assign dma_dram_stride = w3[15:0];
    assign dma_spad_stride = w3[31:16];

    typedef enum logic [0:0] { S_HEAD, S_RUN } state_t;
    state_t state;

    assign dma_start = (state == S_HEAD) && !empty && is_move;
    assign pop       = ((state == S_HEAD) && !empty && !is_move) ||
                       ((state == S_RUN)  && dma_done);
    assign idle      = empty && (state == S_HEAD);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                          state <= S_HEAD;
        else case (state)
            S_HEAD: if (!empty && is_move)   state <= S_RUN;
            S_RUN:  if (dma_done)            state <= S_HEAD;
            default:                         state <= S_HEAD;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            issued  <= '0;
            retired <= '0;
        end else begin
            if (cmd_we && !cmd_full) issued  <= issued  + 32'd1;
            if (pop)                 retired <= retired + 32'd1;
        end
    end

// synthesis translate_off
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && state == S_HEAD && !empty && !is_move)
            $display("[%0t] cmd_dma: unknown command op 0x%02h (discarded)", $time, c_op);
    end
`endif
// synthesis translate_on

endmodule
