`timescale 1ns/1ps
// SIMD unit for every pointwise / reduction op that is not a matmul. Operands
// and results are int4, packed two per byte, one scratchpad bank word per
// access. Narrowing is fused into every op. See docs/vpu.md.

module vpu #(
    parameter int SCRATCHPAD_W = 4,   // scratchpad port width in bytes (one bank word)
    parameter int ADDR_W       = 16,
    parameter int M0_W         = 12,
    parameter int N_W          = 4
) (
    input  logic clk,
    input  logic rst_n,

    input  logic                 vpu_start,
    input  logic [4:0]           vpu_op,
    input  logic [ADDR_W-1:0]    vpu_src0,
    input  logic [ADDR_W-1:0]    vpu_src1,
    input  logic [M0_W+N_W-1:0]  vpu_rq_word,   // {n, m0} literal
    input  logic [ADDR_W-1:0]    vpu_dst,
    input  logic [9:0]           vpu_vlen,      // elements
    output logic                 vpu_busy,
    output logic                 vpu_done,

    output logic                      V_re,
    output logic [ADDR_W-1:0]         V_raddr,
    input  logic [SCRATCHPAD_W*8-1:0] V_rdata,   // valid the cycle after V_re
    output logic                      V_we,
    output logic [ADDR_W-1:0]         V_waddr,
    output logic [SCRATCHPAD_W*8-1:0] V_wdata,
    output logic [SCRATCHPAD_W-1:0]   V_wstrb,
    input  logic                      V_rgnt,
    input  logic                      V_wgnt
);

    localparam int LANES = SCRATCHPAD_W * 2;   // int4 elements per access
    localparam int ACC_W = 32;
    localparam int RQ_W  = 48;
    localparam int Q4_MIN = -8;
    localparam int Q4_MAX =  7;

    initial if (LANES < 2)
        $fatal(1, "vpu: SCRATCHPAD_W (%0d) must be at least 2 bytes", SCRATCHPAD_W);

    // The gaps are retired opcodes, not free encoding space: 2, 4-9, 11-15, and
    // 17, which was VOP_QUANT4 until the MXU began storing int4 itself.
    localparam logic [4:0]
        VOP_DOT     = 5'd0,
        VOP_ADD     = 5'd1,
        VOP_RELU    = 5'd3,
        VOP_REQUANT = 5'd10,
        VOP_DYT     = 5'd16,
        VOP_ARGMAX  = 5'd18;

    // DYT is binary because a normalization always follows a residual add: it
    // is ADD with the odd clip, which is the only form the model uses.
    function automatic logic needs_src1(input logic [4:0] o);
        return (o == VOP_DOT) || (o == VOP_ADD) || (o == VOP_DYT);
    endfunction

    function automatic logic is_reduction(input logic [4:0] o);
        return (o == VOP_DOT) || (o == VOP_ARGMAX);
    endfunction

    // clip((v*m0 + round) >> n). `lo` is -8 for the requantizing ops and -7 for
    // DyT, whose hardtanh is odd.
    function automatic logic signed [3:0] narrow4(input logic signed [ACC_W-1:0] v,
                                                  input logic [M0_W-1:0]         m0,
                                                  input logic [N_W-1:0]          n,
                                                  input int                      lo);
        logic signed [RQ_W-1:0] prod, round, shifted;
        prod    = v * $signed({1'b0, m0});
        round   = (n == 0) ? '0 : (RQ_W'(1) <<< (n - 1));
        shifted = (prod + round) >>> n;
        if      (shifted > Q4_MAX) narrow4 = 4'(signed'(Q4_MAX));
        else if (shifted < lo)     narrow4 = 4'(signed'(lo));
        else                       narrow4 = 4'(shifted);
    endfunction

    typedef enum logic [3:0] {
        S_IDLE, S_RD0, S_RD0D, S_RD1, S_RD1D, S_EXEC, S_WB, S_DONE
    } state_t;
    state_t state, state_n;

    // A denied access freezes the state register and re-drives the same
    // request. V_rdata is only sampled the cycle after a granted read, because
    // the state that samples it is unreachable without the grant.
    wire stalled = (V_re && !V_rgnt) || (V_we && !V_wgnt);

    logic [4:0]        op_r;
    logic [ADDR_W-1:0] dst_r;
    logic [ADDR_W-1:0] p_src0, p_src1, p_dst;
    logic [10:0]       remaining;
    logic [10:0]       p_elem;

    logic [M0_W+N_W-1:0]     rq_word_r;
    logic signed [ACC_W-1:0] acc;

    wire [M0_W-1:0] rq_m0 = rq_word_r[M0_W-1:0];
    wire [N_W-1:0]  rq_n  = rq_word_r[M0_W +: N_W];

    // Every operand and every elementwise result is one word of packed int4, so
    // all three pointers step by the same amount.
    wire [ADDR_W-1:0] chunk_bytes = ADDR_W'(SCRATCHPAD_W);

    wire [10:0] chunk_active = (remaining >= LANES) ? 11'(LANES) : remaining;
    wire        last_chunk   = (remaining <= LANES);

    logic [SCRATCHPAD_W*8-1:0] V_data0, V_data1;

    logic                    lane_active [0:LANES-1];
    logic signed [3:0]       res4    [0:LANES-1];
    logic signed [ACC_W-1:0] red_val [0:LANES-1];

    // ARGMAX folds src0 to (value, index) in a binary tree, so a chunk costs
    // log2(LANES) comparator levels instead of LANES. Level 0 pads up to a
    // power of two with Q4_MIN and every tie goes left; the active lanes are a
    // prefix, so a padding lane can never outrank a real one.
    localparam int TREE_LV = $clog2(LANES);
    localparam int TREE_N  = 1 << TREE_LV;

    logic signed [3:0] fold_val [0:TREE_LV][0:TREE_N-1];
    logic [10:0]       fold_idx [0:TREE_LV][0:TREE_N-1];

    always_comb begin
        for (int lv = 0; lv <= TREE_LV; lv++)
            for (int k = 0; k < TREE_N; k++) begin
                fold_val[lv][k] = 4'(signed'(Q4_MIN));
                fold_idx[lv][k] = '0;
            end
        for (int l = 0; l < TREE_N; l++) begin
            if (l < LANES && 11'(l) < chunk_active)
                fold_val[0][l] = $signed(V_data0[l*4 +: 4]);
            fold_idx[0][l] = 11'(l);
        end
        for (int lv = 1; lv <= TREE_LV; lv++)
            for (int k = 0; k < TREE_N/2; k++)
                if (k < (TREE_N >> lv)) begin
                    if (fold_val[lv-1][2*k] >= fold_val[lv-1][2*k+1]) begin
                        fold_val[lv][k] = fold_val[lv-1][2*k];
                        fold_idx[lv][k] = fold_idx[lv-1][2*k];
                    end else begin
                        fold_val[lv][k] = fold_val[lv-1][2*k+1];
                        fold_idx[lv][k] = fold_idx[lv-1][2*k+1];
                    end
                end
    end

    wire signed [3:0] chunk_max_val = fold_val[TREE_LV][0];
    wire       [10:0] chunk_max_idx = fold_idx[TREE_LV][0];

    logic signed [3:0] a4, b4;

    always_comb begin
        a4 = '0; b4 = '0;
        for (int l = 0; l < LANES; l++) begin
            a4 = V_data0[l*4 +: 4];
            b4 = V_data1[l*4 +: 4];
            lane_active[l] = (l < chunk_active);
            res4[l]    = '0;
            red_val[l] = '0;
            unique case (op_r)
                VOP_ADD:     res4[l] = narrow4(ACC_W'(a4) + ACC_W'(b4), rq_m0, rq_n, Q4_MIN);
                VOP_RELU:    res4[l] = narrow4((a4 > 0) ? ACC_W'(a4) : '0, rq_m0, rq_n, Q4_MIN);
                VOP_REQUANT: res4[l] = narrow4(ACC_W'(a4), rq_m0, rq_n, Q4_MIN);
                VOP_DYT:     res4[l] = narrow4(ACC_W'(a4) + ACC_W'(b4), rq_m0, rq_n, -Q4_MAX);
                VOP_DOT:     red_val[l] = a4 * b4;
                default: ;
            endcase
        end
    end

    // Both reductions land in `acc`: DOT's running sum, ARGMAX's best index so
    // far. ARGMAX carries the best value beside it, one bit wider than an int4
    // so its initial value loses to a vector that is all Q4_MIN. Strictly
    // greater, so the earliest chunk holding the maximum keeps it and the index
    // is the lowest one -- torch.argmax's rule.
    logic signed [ACC_W-1:0] sum_chunk, acc_next;
    logic signed [4:0]       max_val, max_next;
    always_comb begin
        sum_chunk = '0;
        for (int l = 0; l < LANES; l++)
            if (lane_active[l]) sum_chunk += red_val[l];

        max_next = max_val;
        if (op_r == VOP_ARGMAX) begin
            acc_next = acc;
            if (5'(chunk_max_val) > max_val) begin
                acc_next = ACC_W'(p_elem) + ACC_W'(chunk_max_idx);
                max_next = 5'(chunk_max_val);
            end
        end else begin
            acc_next = acc + sum_chunk;
        end
    end

    // Two lanes share a byte, so the strobe is per pair. A tail that leaves a
    // byte half active writes nibble 0 into the other half rather than
    // preserving it, which is why vlen has to be even.
    logic [SCRATCHPAD_W*8-1:0] wb_data;
    logic [SCRATCHPAD_W-1:0]   wb_strb;
    always_comb begin
        wb_data = '0;
        wb_strb = '0;
        for (int l = 0; l < LANES; l++)
            if (lane_active[l]) wb_data[l*4 +: 4] = res4[l];
        for (int b = 0; b < SCRATCHPAD_W; b++)
            if (lane_active[b*2]) wb_strb[b] = 1'b1;
    end

    always_comb begin
        V_re    = 1'b0;
        V_raddr = '0;
        V_we    = 1'b0;
        V_waddr = '0;
        V_wdata = '0;
        V_wstrb = '0;

        unique case (state)
            S_RD0: begin V_re = 1'b1; V_raddr = p_src0; end
            S_RD1: begin V_re = 1'b1; V_raddr = p_src1; end
            S_EXEC: if (!is_reduction(op_r)) begin
                V_we    = 1'b1;
                V_waddr = p_dst;
                V_wdata = wb_data;
                V_wstrb = wb_strb;
            end
            // The reduction's int32 scalar. It is the one result wider than a
            // nibble, and it needs a whole word to itself.
            S_WB: begin
                V_we               = 1'b1;
                V_waddr            = dst_r;
                V_wdata[ACC_W-1:0] = acc;
                V_wstrb            = '1;
            end
            default: ;
        endcase
    end

    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE: if (vpu_start) begin
                        if (vpu_vlen == 0) state_n = S_DONE;
                        else               state_n = S_RD0;
                    end
            S_RD0:  state_n = S_RD0D;
            S_RD0D: begin
                        if (needs_src1(op_r)) state_n = S_RD1;
                        else                  state_n = S_EXEC;
                    end
            S_RD1:  state_n = S_RD1D;
            S_RD1D: state_n = S_EXEC;
            S_EXEC: begin
                        if (last_chunk) begin
                            if (is_reduction(op_r)) state_n = S_WB;
                            else                    state_n = S_DONE;
                        end else state_n = S_RD0;
                    end
            S_WB:   state_n = S_DONE;
            S_DONE: state_n = S_IDLE;
            default: state_n = S_IDLE;
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            op_r      <= '0;
            dst_r     <= '0;
            p_src0    <= '0;
            p_src1    <= '0;
            p_dst     <= '0;
            rq_word_r <= '0;
            remaining <= '0;
            acc       <= '0;
            V_data0   <= '0;
            V_data1   <= '0;
            p_elem    <= '0;
            max_val   <= 5'(signed'(Q4_MIN - 1));
        end else if (!stalled) begin
            state <= state_n;

            unique case (state)
                S_IDLE: if (vpu_start) begin
                    op_r      <= vpu_op;
                    dst_r     <= vpu_dst;
                    p_src0    <= vpu_src0;
                    p_src1    <= vpu_src1;
                    p_dst     <= vpu_dst;
                    rq_word_r <= vpu_rq_word;
                    remaining <= {1'b0, vpu_vlen};
                    acc       <= '0;
                    p_elem    <= '0;
                    max_val   <= 5'(signed'(Q4_MIN - 1));
                end

                S_RD0D: V_data0 <= V_rdata;
                S_RD1D: V_data1 <= V_rdata;
                S_EXEC: begin
                    if (is_reduction(op_r)) begin
                        acc     <= acc_next;
                        max_val <= max_next;
                    end
                    p_src0    <= p_src0 + chunk_bytes;
                    p_src1    <= p_src1 + chunk_bytes;
                    p_dst     <= p_dst  + chunk_bytes;
                    p_elem    <= p_elem + chunk_active;
                    remaining <= remaining - chunk_active;
                end

                default: ;
            endcase
        end
    end

    assign vpu_busy = (state != S_IDLE);
    assign vpu_done = (state == S_DONE);

endmodule
