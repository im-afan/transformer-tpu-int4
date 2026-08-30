`timescale 1ns/1ps
// VPU macro-op front end: command queue + decode. Same shape as cmd_mxu.sv.
// See docs/vpu.md for the retired VPU_GEOM (0x02) / VECMATMUL history.

module cmd_vpu #(
    parameter int ADDR_W = 16,
    parameter int M0_W   = 12,
    parameter int N_W    = 4,
    parameter int DEPTH  = 8
) (
    input  logic clk,
    input  logic rst_n,

    // ---- producer side ------------------------------------------------------
    input  logic         cmd_we,
    input  logic [127:0] cmd_wdata,
    output logic         cmd_full,

    // ---- VPU dispatch (vpu.sv) ----------------------------------------------
    output logic                 vpu_start,
    output logic [4:0]           vpu_op,
    output logic [ADDR_W-1:0]    vpu_src0,
    output logic [ADDR_W-1:0]    vpu_src1,
    output logic [M0_W+N_W-1:0]  vpu_rq_word,
    output logic [ADDR_W-1:0]    vpu_dst,
    output logic [9:0]           vpu_vlen,
    input  logic                 vpu_done,

    // ---- status -------------------------------------------------------------
    output logic [31:0] issued,
    output logic [31:0] retired,
    output logic [15:0] level,
    output logic        idle
);

    localparam logic [7:0] VPU_CMD_OP = 8'h01;   // 8'h02 (GEOM) is retired

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

    wire is_op = (c_op == VPU_CMD_OP);

    // ---- per-op operands ----------------------------------------------------
    assign vpu_op      = w0[12:8];
    assign vpu_dst     = w0[31:16];
    assign vpu_src0    = w1[15:0];
    assign vpu_src1    = w1[31:16];
    assign vpu_vlen    = w2[9:0];
    assign vpu_rq_word = w2[16 +: (M0_W+N_W)];

    typedef enum logic [0:0] { S_HEAD, S_RUN } state_t;
    state_t state;

    assign vpu_start = (state == S_HEAD) && !empty && is_op;
    assign pop       = ((state == S_HEAD) && !empty && !is_op) ||
                       ((state == S_RUN)  && vpu_done);

    assign idle    = empty && (state == S_HEAD);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_HEAD;
        end else begin
            case (state)
                S_HEAD: if (!empty && is_op) state <= S_RUN;
                S_RUN: if (vpu_done) state <= S_HEAD;
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
        if (rst_n && state == S_HEAD && !empty && !is_op)
            $display("[%0t] cmd_vpu: unknown command op 0x%02h (discarded)", $time, c_op);
    end
`endif
// synthesis translate_on

endmodule
