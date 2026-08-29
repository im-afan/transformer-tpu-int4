`timescale 1ns/1ps
// N x N output-stationary int4 systolic array. One dispatch computes
// C[N][N] = A[N][len] @ B (or @ B' when `transpose`), requantized to int4.
// See docs/mxu.md.

module mxu #(
    parameter int N      = 8,    // array size; every operand word is N*4 bits
    parameter int ADDR_W = 16,   // scratchpad byte address
    parameter int M0_W   = 12,   // requant multiplier width
    parameter int N_W    = 4,    // requant shift width
    parameter int ACC_W  = 32    // per-PE accumulator
) (
    input  logic clk,
    input  logic rst_n,

    // ---- dispatch -----------------------------------------------------------
    input  logic                start,
    input  logic                transpose,    // B is transposed in the matmul
    input  logic                accumulate,   // add the existing C
    input  logic [15:0]         len,          // contraction length, int4 elements
    input  logic [ADDR_W-1:0]   a_base,
    input  logic [ADDR_W-1:0]   a_stride,     // bytes between A rows
    input  logic [ADDR_W-1:0]   b_base,
    input  logic [ADDR_W-1:0]   b_stride,     // bytes between B rows
    input  logic [ADDR_W-1:0]   c_base,
    input  logic [ADDR_W-1:0]   c_stride,     // bytes between C rows
    input  logic [M0_W+N_W-1:0] rq_word,      // {n, m0}
    output logic                busy,
    output logic                done,

    // ---- scratchpad ---------------------------------------------------------
    output logic              A_re,
    output logic [ADDR_W-1:0] A_addr,
    input  logic [N*4-1:0]    A_rdata,
    input  logic              A_gnt,

    output logic              B_re,
    output logic [ADDR_W-1:0] B_addr,
    input  logic [N*4-1:0]    B_rdata,
    input  logic              B_gnt,

    output logic              C_en,
    output logic              C_we,
    output logic [ADDR_W-1:0] C_addr,
    output logic [N*4-1:0]    C_wdata,
    input  logic [N*4-1:0]    C_rdata,
    input  logic              C_gnt
);

    localparam int LOGN       = (N > 1) ? $clog2(N) : 1;
    localparam int WORD_BYTES = N / 2;
    localparam int CNT_W      = 20;
    localparam int RQ_W       = ACC_W + M0_W + 4;
    localparam int QMIN       = -8;
    localparam int QMAX       =  7;

    initial if (N < 2 || (N & (N - 1)) != 0)
        $fatal(1, "mxu: N (%0d) must be a power of two, at least 2", N);

    // int32 -> int4: clip((acc*m0 + round) >> n).
    function automatic logic signed [3:0] requant4(input logic signed [ACC_W-1:0] acc_i,
                                                   input logic [M0_W-1:0]         m0,
                                                   input logic [N_W-1:0]          n);
        logic signed [RQ_W-1:0] product, bias, shifted;
        product = acc_i * $signed({1'b0, m0});
        bias    = (n == 0) ? '0 : (RQ_W'(1) <<< (n - 1));
        shifted = (product + bias) >>> n;
        if      (shifted > QMAX) requant4 = 4'(signed'(QMAX));
        else if (shifted < QMIN) requant4 = 4'(signed'(QMIN));
        else                     requant4 = 4'(shifted);
    endfunction

    function automatic logic signed [3:0] clip4(input logic signed [5:0] v);
        if      (v > QMAX) clip4 = 4'(signed'(QMAX));
        else if (v < QMIN) clip4 = 4'(signed'(QMIN));
        else               clip4 = 4'(v);
    endfunction

    localparam int S_IDLE = 0, S_STREAM = 1, S_DRAIN_RD = 2, S_DRAIN_WR = 3, S_DONE = 4;
    logic [2:0] state;

    // ---- latched dispatch ---------------------------------------------------
    logic                transpose_q, acc_q;
    logic [15:0]         len_q;
    logic [ADDR_W-1:0]   a_base_q, b_base_q, c_base_q;
    logic [ADDR_W-1:0]   a_stride_q, b_stride_q, c_stride_q;
    logic [M0_W-1:0]     rq_m0;
    logic [N_W-1:0]      rq_n;
    logic [CNT_W-1:0]    a_reads, b_reads, sc_last;

    // Zero stride means the densely packed default.
    wire [ADDR_W-1:0] len_bytes    = ADDR_W'(len >> 1);
    wire [ADDR_W-1:0] a_stride_sel = (a_stride != '0) ? a_stride : len_bytes;
    wire [ADDR_W-1:0] b_stride_sel = (b_stride != '0) ? b_stride
                                   : (transpose ? len_bytes : ADDR_W'(WORD_BYTES));
    wire [ADDR_W-1:0] c_stride_sel = (c_stride != '0) ? c_stride : ADDR_W'(WORD_BYTES);

    wire [CNT_W-1:0] chunks_sel = (CNT_W'(len) + CNT_W'(N-1)) >> LOGN;

    // ---- operand feed -------------------------------------------------------
    logic [CNT_W-1:0]  iq;                     // operand words requested
    logic [CNT_W-1:0]  sc;                     // array steps taken
    logic              arrive;                 // the bus carries step `sc`'s words
    logic [ADDR_W-1:0] a_off_row, a_off_chunk;
    logic [ADDR_W-1:0] b_off_row, b_off_chunk;

    wire a_req    = (state == S_STREAM) && (iq < a_reads);
    wire b_req    = (state == S_STREAM) && (iq < b_reads);
    wire issue_ok = (!a_req || A_gnt) && (!b_req || B_gnt);
    wire issue_go = (a_req || b_req) && issue_ok;
    wire row_wrap = (iq[LOGN-1:0] == LOGN'(N-1));

    wire issues_done = (iq >= a_reads) && (iq >= b_reads);
    wire array_en    = (state == S_STREAM) && (arrive || issues_done);

    assign A_re   = a_req;
    assign B_re   = b_req;
    assign A_addr = a_base_q + a_off_row + a_off_chunk;
    assign B_addr = b_base_q + b_off_row + b_off_chunk;

    // ---- edge registers -----------------------------------------------------
    // A always arrives as a chunk of N contraction elements for one array row,
    // round-robin, so the round-robin is the skew. B does the same when
    // transposed; otherwise a whole B row lands at once and the top edge needs a
    // triangular skew chain.
    logic [N*4-1:0] arow_buf [0:N-1];
    logic [N*4-1:0] bcol_buf [0:N-1];
    logic [N*4-1:0] bskew    [0:N-1];
    logic [N-1:0]   vchain;

    wire feed_valid = (sc >= CNT_W'(1)) && (sc <= CNT_W'(len_q));

    logic signed [3:0] a_edge [0:N-1];
    logic signed [3:0] b_edge [0:N-1];
    logic              v_edge [0:N-1];

    always_comb begin
        for (int i = 0; i < N; i++) begin
            a_edge[i] = 4'(arow_buf[i]);
            b_edge[i] = transpose_q ? 4'(bcol_buf[i]) : bskew[i][i*4 +: 4];
            v_edge[i] = (i == 0) ? feed_valid : vchain[i-1];
        end
    end

    // ---- PE array -----------------------------------------------------------
    logic signed [3:0]       pe_a [0:N-1][0:N-1];
    logic signed [3:0]       pe_b [0:N-1][0:N-1];
    logic                    pe_v [0:N-1][0:N-1];
    logic signed [ACC_W-1:0] acc  [0:N-1][0:N-1];

    logic signed [3:0] in_a, in_b;
    logic signed [7:0] in_prod;
    logic              in_v;

    // ---- writeback ----------------------------------------------------------
    logic [LOGN-1:0] drow;

    logic signed [3:0] rq_elem, old_elem;
    always_comb begin
        C_wdata = '0;
        for (int j = 0; j < N; j++) begin
            rq_elem  = requant4(acc[drow][j], rq_m0, rq_n);
            old_elem = acc_q ? $signed(C_rdata[j*4 +: 4]) : 4'sd0;
            C_wdata[j*4 +: 4] = clip4(6'(rq_elem) + 6'(old_elem));
        end
    end

    assign C_en   = (state == S_DRAIN_RD) || (state == S_DRAIN_WR);
    assign C_we   = (state == S_DRAIN_WR);
    assign C_addr = c_base_q + ADDR_W'(drow) * c_stride_q;

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            iq          <= '0;
            sc          <= '0;
            arrive      <= 1'b0;
            drow        <= '0;
            vchain      <= '0;
            transpose_q <= 1'b0;
            acc_q       <= 1'b0;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) pe_v[i][j] <= 1'b0;
        end else begin
            arrive <= issue_go;

            case (state)
                S_IDLE: if (start) begin
                    transpose_q <= transpose;
                    acc_q       <= accumulate;
                    len_q       <= len;
                    a_base_q    <= a_base;
                    b_base_q    <= b_base;
                    c_base_q    <= c_base;
                    a_stride_q  <= a_stride_sel;
                    b_stride_q  <= b_stride_sel;
                    c_stride_q  <= c_stride_sel;
                    rq_m0       <= rq_word[M0_W-1:0];
                    rq_n        <= rq_word[M0_W +: N_W];
                    a_reads     <= chunks_sel << LOGN;
                    b_reads     <= transpose ? (chunks_sel << LOGN) : CNT_W'(len);
                    sc_last     <= CNT_W'(len) + CNT_W'(2*N - 2);

                    iq          <= '0;
                    sc          <= '0;
                    arrive      <= 1'b0;
                    drow        <= '0;
                    vchain      <= '0;
                    a_off_row   <= '0;
                    a_off_chunk <= '0;
                    b_off_row   <= '0;
                    b_off_chunk <= '0;
                    for (int i = 0; i < N; i++)
                        for (int j = 0; j < N; j++) begin
                            acc[i][j]  <= '0;
                            pe_v[i][j] <= 1'b0;
                        end

                    state <= (len == 16'd0) ? S_DONE : S_STREAM;
                end

                S_STREAM: begin
                    // Operand fetch. A denied port re-presents; nothing advances.
                    if (issue_go) begin
                        iq <= iq + CNT_W'(1);
                        if (a_req) begin
                            if (row_wrap) begin
                                a_off_row   <= '0;
                                a_off_chunk <= a_off_chunk + ADDR_W'(WORD_BYTES);
                            end else begin
                                a_off_row   <= a_off_row + a_stride_q;
                            end
                        end
                        if (b_req) begin
                            if (transpose_q && row_wrap) begin
                                b_off_row   <= '0;
                                b_off_chunk <= b_off_chunk + ADDR_W'(WORD_BYTES);
                            end else begin
                                b_off_row   <= b_off_row + b_stride_q;
                            end
                        end
                    end

                    if (array_en) begin
                        sc     <= sc + CNT_W'(1);
                        vchain <= (vchain << 1) | N'(feed_valid);

                        // Edge buffers. A load beats the shift: the clock a row
                        // takes its next chunk is the clock it runs out.
                        for (int i = 0; i < N; i++) begin
                            if (sc < a_reads && sc[LOGN-1:0] == LOGN'(i))
                                arow_buf[i] <= A_rdata;
                            else if (v_edge[i])
                                arow_buf[i] <= arow_buf[i] >> 4;
                        end

                        if (transpose_q) begin
                            for (int j = 0; j < N; j++) begin
                                if (sc < b_reads && sc[LOGN-1:0] == LOGN'(j))
                                    bcol_buf[j] <= B_rdata;
                                else if (v_edge[j])
                                    bcol_buf[j] <= bcol_buf[j] >> 4;
                            end
                        end else begin
                            bskew[0] <= B_rdata;
                            for (int s = 1; s < N; s++) bskew[s] <= bskew[s-1];
                        end

                        for (int i = 0; i < N; i++)
                            for (int j = 0; j < N; j++) begin
                                in_a = (j == 0) ? a_edge[i] : pe_a[i][j-1];
                                in_v = (j == 0) ? v_edge[i] : pe_v[i][j-1];
                                in_b = (i == 0) ? b_edge[j] : pe_b[i-1][j];
                                in_prod = in_a * in_b;

                                pe_a[i][j] <= in_a;
                                pe_v[i][j] <= in_v;
                                pe_b[i][j] <= in_b;
                                if (in_v)
                                    acc[i][j] <= acc[i][j]
                                               + {{(ACC_W-8){in_prod[7]}}, in_prod};
                            end

                        if (sc == sc_last) state <= acc_q ? S_DRAIN_RD : S_DRAIN_WR;
                    end
                end

                S_DRAIN_RD: if (C_gnt) state <= S_DRAIN_WR;

                S_DRAIN_WR: if (C_gnt) begin
                    drow <= drow + LOGN'(1);
                    if (drow == LOGN'(N-1)) state <= S_DONE;
                    else                    state <= acc_q ? S_DRAIN_RD : S_DRAIN_WR;
                end

                S_DONE: state <= S_IDLE;

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
