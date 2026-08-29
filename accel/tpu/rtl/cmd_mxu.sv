`timescale 1ns/1ps
// MXU command front end: 128-bit queue, decode, issue-and-wait against the
// array. MXU_GEOM latches the strides and the contraction length and retires in
// one clock; MXU_MM runs one matmul. See docs/mxu.md.

module cmd_mxu #(
    parameter int ADDR_W = 16,
    parameter int M0_W   = 12,
    parameter int N_W    = 4,
    parameter int DEPTH  = 8
) (
    input  logic clk,
    input  logic rst_n,

    input  logic         cmd_we,
    input  logic [127:0] cmd_wdata,
    output logic         cmd_full,

    output logic                mxu_start,
    output logic                mxu_transpose,
    output logic                mxu_accumulate,
    output logic [15:0]         mxu_len,
    output logic [ADDR_W-1:0]   mxu_a_base,
    output logic [ADDR_W-1:0]   mxu_a_stride,
    output logic [ADDR_W-1:0]   mxu_b_base,
    output logic [ADDR_W-1:0]   mxu_b_stride,
    output logic [ADDR_W-1:0]   mxu_c_base,
    output logic [ADDR_W-1:0]   mxu_c_stride,
    output logic [M0_W+N_W-1:0] mxu_rq_word,
    input  logic                mxu_done,

    output logic [31:0] issued,
    output logic [31:0] retired,
    output logic [15:0] level,
    output logic        idle
);

    localparam logic [7:0] MXU_GEOM = 8'h01,
                           MXU_MM   = 8'h02;

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

    wire is_geom = (c_op == MXU_GEOM);
    wire is_mm   = (c_op == MXU_MM);

    logic [ADDR_W-1:0] g_astride, g_bstride, g_cstride;
    logic [15:0]       g_len;

    assign mxu_a_stride = g_astride;
    assign mxu_b_stride = g_bstride;
    assign mxu_c_stride = g_cstride;
    assign mxu_len      = g_len;

    assign mxu_c_base     = w0[31:16];
    assign mxu_accumulate = w0[8];
    assign mxu_transpose  = w0[9];
    assign mxu_a_base     = w1[15:0];
    assign mxu_b_base     = w1[31:16];
    assign mxu_rq_word    = w2[M0_W+N_W-1:0];

    typedef enum logic [0:0] { S_HEAD, S_RUN } state_t;
    state_t state;

    assign mxu_start = (state == S_HEAD) && !empty && is_mm;
    assign pop       = ((state == S_HEAD) && !empty && !is_mm) ||
                       ((state == S_RUN)  && mxu_done);
    assign idle      = empty && (state == S_HEAD);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_HEAD;
            g_astride <= '0;
            g_bstride <= '0;
            g_cstride <= '0;
            g_len     <= '0;
        end else begin
            case (state)
                S_HEAD: if (!empty) begin
                    if (is_geom) begin
                        g_astride <= w0[31:16];
                        g_bstride <= w1[15:0];
                        g_cstride <= w1[31:16];
                        g_len     <= w2[15:0];
                    end else if (is_mm) begin
                        state <= S_RUN;
                    end
                end
                S_RUN: if (mxu_done) state <= S_HEAD;
                default: state <= S_HEAD;
            endcase
        end
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
        if (rst_n && state == S_HEAD && !empty && !is_geom && !is_mm)
            $display("[%0t] cmd_mxu: unknown command op 0x%02h (discarded)", $time, c_op);
    end
`endif
// synthesis translate_on

endmodule
