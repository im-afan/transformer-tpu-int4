`timescale 1ns/1ps
// Self-checking testbench for rtl/mxu.sv.
//
// Models the three scratchpad ports (A, B, C — all one N*4-bit word per access,
// valid the cycle after a granted request) and checks the output-stationary
// array against an integer reference. Covers both B layouts, contraction
// lengths above/below/not-a-multiple-of N, explicit strides, accumulate, the
// requant clip, and a randomly denied operand port.
//
// Run:  make TEST=mxu sim

module mxu_tb;

    localparam int N          = 4;
    localparam int ADDR_W     = 16;
    localparam int M0_W       = 12;
    localparam int N_W        = 4;
    localparam int WORD_BYTES = N / 2;
    localparam int MAXLEN     = 32;

    localparam logic [ADDR_W-1:0] A_BASE = 16'h1000;
    localparam logic [ADDR_W-1:0] B_BASE = 16'h2000;
    localparam logic [ADDR_W-1:0] C_BASE = 16'h3000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---- DUT interface ------------------------------------------------------
    logic                transpose, accumulate, start, busy, done;
    logic [15:0]         len;
    logic [ADDR_W-1:0]   a_base, a_stride, b_base, b_stride, c_base, c_stride;
    logic [M0_W+N_W-1:0] rq_word;

    logic              A_re, A_gnt;
    logic [ADDR_W-1:0] A_addr;
    logic [N*4-1:0]    A_rdata;
    logic              B_re, B_gnt;
    logic [ADDR_W-1:0] B_addr;
    logic [N*4-1:0]    B_rdata;
    logic              C_en, C_we, C_gnt;
    logic [ADDR_W-1:0] C_addr;
    logic [N*4-1:0]    C_wdata, C_rdata;

    mxu #(.N(N), .ADDR_W(ADDR_W), .M0_W(M0_W), .N_W(N_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .transpose(transpose), .accumulate(accumulate), .len(len),
        .a_base(a_base), .a_stride(a_stride),
        .b_base(b_base), .b_stride(b_stride),
        .c_base(c_base), .c_stride(c_stride),
        .rq_word(rq_word), .busy(busy), .done(done),
        .A_re(A_re), .A_addr(A_addr), .A_rdata(A_rdata), .A_gnt(A_gnt),
        .B_re(B_re), .B_addr(B_addr), .B_rdata(B_rdata), .B_gnt(B_gnt),
        .C_en(C_en), .C_we(C_we), .C_addr(C_addr),
        .C_wdata(C_wdata), .C_rdata(C_rdata), .C_gnt(C_gnt)
    );

    // ---- Scratchpad model ---------------------------------------------------
    localparam int MEM_SZ = 1 << ADDR_W;
    logic [7:0] mem [0:MEM_SZ-1];

    // Denial injector: the array must re-present and lose nothing.
    logic deny_enable = 1'b0;
    logic deny_a, deny_b;
    always_ff @(posedge clk) begin
        deny_a <= deny_enable && ($random % 4 == 0);
        deny_b <= deny_enable && ($random % 4 == 1);
    end

    assign A_gnt = A_re && !deny_a;
    assign B_gnt = B_re && !deny_b;
    assign C_gnt = C_en;

    logic [N*4-1:0] a_rd_r, b_rd_r, c_rd_r;

    always_ff @(posedge clk) begin
        if (A_re && A_gnt)
            for (int b = 0; b < WORD_BYTES; b++) a_rd_r[b*8 +: 8] <= mem[A_addr + b];
        if (B_re && B_gnt)
            for (int b = 0; b < WORD_BYTES; b++) b_rd_r[b*8 +: 8] <= mem[B_addr + b];
        if (C_en && !C_we && C_gnt)
            for (int b = 0; b < WORD_BYTES; b++) c_rd_r[b*8 +: 8] <= mem[C_addr + b];
        if (C_en && C_we && C_gnt)
            for (int b = 0; b < WORD_BYTES; b++) mem[C_addr + b] <= C_wdata[b*8 +: 8];
    end

    assign A_rdata = a_rd_r;
    assign B_rdata = b_rd_r;
    assign C_rdata = c_rd_r;

    // ---- Reference ----------------------------------------------------------
    int  aval [0:N-1][0:MAXLEN-1];
    int  bval [0:MAXLEN-1][0:N-1];   // always the mathematical B[k][j]
    int  cpre [0:N-1][0:N-1];
    int  errors = 0, checks = 0;

    task automatic put_nib(input [ADDR_W-1:0] base, input int idx, input int v);
        mem[base + (idx/2)][(idx%2)*4 +: 4] = v[3:0];
    endtask

    function automatic int get_nib(input [ADDR_W-1:0] base, input int idx);
        return $signed(mem[base + (idx/2)][(idx%2)*4 +: 4]);
    endfunction

    function automatic int rnd4();
        return int'($unsigned($random) % 16) - 8;
    endfunction

    function automatic int ref_requant(input int acc_v, input int m0, input int n);
        longint prod, bias, sh;
        prod = longint'(acc_v) * longint'(m0);
        bias = (n == 0) ? 0 : (longint'(1) << (n - 1));
        sh   = (prod + bias) >>> n;
        if (sh >  7) return  7;
        if (sh < -8) return -8;
        return int'(sh);
    endfunction

    function automatic int clip4(input int v);
        if (v >  7) return  7;
        if (v < -8) return -8;
        return v;
    endfunction

    function automatic int ref_dot(input int i, input int j, input int L);
        int s;
        s = 0;
        for (int k = 0; k < L; k++) s += aval[i][k] * bval[k][j];
        return s;
    endfunction

    // A is always [N][len], contraction contiguous.
    task automatic gen_a(input int L, input [ADDR_W-1:0] stride);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < L; k++) begin
                aval[i][k] = rnd4();
                put_nib(A_BASE + stride * ADDR_W'(i), k, aval[i][k]);
            end
        aval[0][0] = -8; put_nib(A_BASE, 0, -8);
        if (L > 1) begin aval[0][1] = 7; put_nib(A_BASE, 1, 7); end
    endtask

    // transpose = 0: stored [len][N], one B row per contraction step.
    // transpose = 1: stored [N][len], one contiguous run per output column.
    task automatic gen_b(input int L, input logic tr, input [ADDR_W-1:0] stride);
        for (int k = 0; k < L; k++)
            for (int j = 0; j < N; j++) begin
                bval[k][j] = rnd4();
                if (tr) put_nib(B_BASE + stride * ADDR_W'(j), k, bval[k][j]);
                else    put_nib(B_BASE + stride * ADDR_W'(k), j, bval[k][j]);
            end
        bval[0][0] = -8;
        if (tr) put_nib(B_BASE, 0, -8);
        else    put_nib(B_BASE, 0, -8);
    endtask

    task automatic seed_c(input [ADDR_W-1:0] stride);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                cpre[i][j] = rnd4();
                put_nib(C_BASE + stride * ADDR_W'(i), j, cpre[i][j]);
            end
    endtask

    task automatic dispatch(input int L, input logic tr, input logic acc,
                            input [ADDR_W-1:0] as, input [ADDR_W-1:0] bs,
                            input [ADDR_W-1:0] cs, input int m0, input int n);
        @(negedge clk);
        len        = L[15:0];
        transpose  = tr;
        accumulate = acc;
        a_base     = A_BASE;  a_stride = as;
        b_base     = B_BASE;  b_stride = bs;
        c_base     = C_BASE;  c_stride = cs;
        rq_word    = (n[N_W-1:0] << M0_W) | m0[M0_W-1:0];
        start      = 1'b1;
        @(negedge clk);
        start = 1'b0;
        do @(negedge clk); while (!done);
        @(negedge clk);
    endtask

    task automatic check_c(input int L, input logic acc, input [ADDR_W-1:0] cs,
                           input int m0, input int n, input string tag);
        int got, exp;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                exp = ref_requant(ref_dot(i, j, L), m0, n);
                if (acc) exp = clip4(exp + cpre[i][j]);
                got = get_nib(C_BASE + cs * ADDR_W'(i), j);
                checks++;
                if (got !== exp) begin
                    errors++;
                    if (errors < 20)
                        $display("  FAIL %-14s C[%0d][%0d]: got %0d exp %0d (dot %0d)",
                                 tag, i, j, got, exp, ref_dot(i, j, L));
                end
            end
        $display("[%-14s] len=%0d  (errors so far: %0d)", tag, L, errors);
    endtask

    // One full case: generate operands, run, compare.
    task automatic test_case(input int L, input logic tr, input logic acc,
                             input int m0, input int n, input string tag);
        logic [ADDR_W-1:0] as, bs, cs;
        as = ADDR_W'((L + 1) / 2);
        bs = tr ? ADDR_W'((L + 1) / 2) : ADDR_W'(WORD_BYTES);
        cs = ADDR_W'(WORD_BYTES);
        gen_a(L, as);
        gen_b(L, tr, bs);
        if (acc) seed_c(cs);
        dispatch(L, tr, acc, as, bs, cs, m0, n);
        check_c(L, acc, cs, m0, n, tag);
    endtask

    // The same with every operand sitting inside a wider matrix.
    task automatic test_strided(input int L, input logic tr, input string tag);
        logic [ADDR_W-1:0] as, bs, cs;
        as = ADDR_W'((L + 1) / 2 + 3);
        bs = tr ? ADDR_W'((L + 1) / 2 + 3) : ADDR_W'(WORD_BYTES + 2);
        cs = ADDR_W'(WORD_BYTES + 5);
        gen_a(L, as);
        gen_b(L, tr, bs);
        dispatch(L, tr, 1'b0, as, bs, cs, 8, 6);
        check_c(L, 1'b0, cs, 8, 6, tag);
    endtask

    // ---- Stimulus -----------------------------------------------------------
    initial begin
        start = 0; transpose = 0; accumulate = 0; len = 0;
        a_base = 0; a_stride = 0; b_base = 0; b_stride = 0;
        c_base = 0; c_stride = 0; rq_word = 0;
        for (int i = 0; i < MEM_SZ; i++) mem[i] = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== MXU testbench (%0dx%0d array) ====", N, N);

        // m0/n chosen so the products land across the int4 grid rather than
        // pinned at the clip: |dot| <= len*64.
        test_case(N,        1'b0, 1'b0, 8,  6, "MM-len4");
        test_case(2*N,      1'b0, 1'b0, 8,  7, "MM-len8");
        test_case(3*N,      1'b0, 1'b0, 8,  8, "MM-len12");
        test_case(N + 2,    1'b0, 1'b0, 8,  7, "MM-len6");
        test_case(2,        1'b0, 1'b0, 8,  5, "MM-len2");

        test_case(N,        1'b1, 1'b0, 8,  6, "MMT-len4");
        test_case(2*N,      1'b1, 1'b0, 8,  7, "MMT-len8");
        test_case(N + 2,    1'b1, 1'b0, 8,  7, "MMT-len6");

        test_case(2*N,      1'b0, 1'b1, 8,  8, "ACC-len8");
        test_case(2*N,      1'b1, 1'b1, 8,  8, "ACCT-len8");

        test_case(2*N,      1'b0, 1'b0, 512, 4, "CLIP-len8");   // saturates

        test_strided(2*N,   1'b0, "STR-len8");
        test_strided(2*N,   1'b1, "STRT-len8");

        // Zero strides must resolve to the dense defaults.
        gen_a(2*N, ADDR_W'(N));
        gen_b(2*N, 1'b0, ADDR_W'(WORD_BYTES));
        dispatch(2*N, 1'b0, 1'b0, '0, '0, '0, 8, 7);
        check_c(2*N, 1'b0, ADDR_W'(WORD_BYTES), 8, 7, "DEFSTRIDE");

        // Denied operand reads: the feed must stall, not drop a word.
        deny_enable = 1'b1;
        test_case(3*N, 1'b0, 1'b0, 8, 8, "STALL");
        test_case(3*N, 1'b1, 1'b0, 8, 8, "STALL-T");
        deny_enable = 1'b0;

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("MXU: ALL TESTS PASSED");
        else             $display("MXU: FAILED (%0d errors)", errors);
        $finish;
    end

    initial begin
        #2000000;
        $display("MXU: TIMEOUT — DUT did not complete");
        $fatal(1);
    end

endmodule
