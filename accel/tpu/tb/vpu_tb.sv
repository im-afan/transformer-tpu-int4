`timescale 1ns/1ps
// Self-checking testbench for rtl/vpu.sv. Every operand and every elementwise
// result is packed int4, so the scratchpad model is byte-addressed and the
// helpers below read and write nibbles. DYT is ADD with the odd clip, so its
// reference is the same expression. DOT and ARGMAX are the two reductions and
// write an int32 word instead. See docs/vpu.md.
//
//   iverilog -g2012 -o vpu_tb.vvp vpu_tb.sv ../rtl/vpu.sv && vvp vpu_tb.vvp

module vpu_tb;

    localparam int SCRATCHPAD_W = 4;                  // one bank word at N=8
    localparam int ADDR_W       = 16;
    localparam int M0_W         = 12;
    localparam int N_W          = 4;
    localparam int LANES        = SCRATCHPAD_W * 2;   // int4 elements per access

    localparam logic [4:0]
        VOP_DOT = 5'd0, VOP_ADD = 5'd1, VOP_RELU = 5'd3,
        VOP_REQUANT = 5'd10, VOP_DYT = 5'd16, VOP_ARGMAX = 5'd18;

    localparam logic [ADDR_W-1:0] A_ADDR = 16'h1000,
                                  B_ADDR = 16'h2000,
                                  D_ADDR = 16'h4000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic                      vpu_start;
    logic [4:0]                vpu_op;
    logic [ADDR_W-1:0]         vpu_src0, vpu_src1, vpu_dst;
    logic [M0_W+N_W-1:0]       vpu_rq_word;
    logic [9:0]                vpu_vlen;
    logic                      vpu_busy, vpu_done;

    logic                      V_re;
    logic [ADDR_W-1:0]         V_raddr;
    logic [SCRATCHPAD_W*8-1:0] V_rdata;
    logic                      V_we;
    logic [ADDR_W-1:0]         V_waddr;
    logic [SCRATCHPAD_W*8-1:0] V_wdata;
    logic [SCRATCHPAD_W-1:0]   V_wstrb;

    vpu #(
        .SCRATCHPAD_W(SCRATCHPAD_W), .ADDR_W(ADDR_W),
        .M0_W(M0_W), .N_W(N_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .vpu_start(vpu_start), .vpu_op(vpu_op),
        .vpu_src0(vpu_src0), .vpu_src1(vpu_src1),
        .vpu_rq_word(vpu_rq_word), .vpu_dst(vpu_dst),
        .vpu_vlen(vpu_vlen),
        .vpu_busy(vpu_busy), .vpu_done(vpu_done),
        .V_re(V_re), .V_raddr(V_raddr), .V_rdata(V_rdata),
        .V_we(V_we), .V_waddr(V_waddr), .V_wdata(V_wdata), .V_wstrb(V_wstrb),
        .V_rgnt(1'b1), .V_wgnt(1'b1)
    );

    localparam int MEM_SZ = 1 << ADDR_W;
    logic [7:0] mem [0:MEM_SZ-1];

    logic [SCRATCHPAD_W*8-1:0] rdata_r;
    always_ff @(posedge clk) begin
        if (V_re)
            for (int k = 0; k < SCRATCHPAD_W; k++)
                rdata_r[k*8 +: 8] <= mem[V_raddr + k];
    end
    assign V_rdata = rdata_r;

    always_ff @(posedge clk) begin
        if (V_we)
            for (int k = 0; k < SCRATCHPAD_W; k++)
                if (V_wstrb[k]) mem[V_waddr + k] <= V_wdata[k*8 +: 8];
    end

    // ---- packed int4 access, two elements per byte, low nibble first --------
    task automatic put4(input [ADDR_W-1:0] base, input int idx, input int v);
        if (idx[0] == 0) mem[base + idx/2][3:0] = v[3:0];
        else             mem[base + idx/2][7:4] = v[3:0];
    endtask

    function automatic int get4(input [ADDR_W-1:0] base, input int idx);
        logic signed [3:0] nib;
        nib = (idx[0] == 0) ? mem[base + idx/2][3:0] : mem[base + idx/2][7:4];
        return int'(nib);
    endfunction

    function automatic int get32(input [ADDR_W-1:0] a);
        return int'({mem[a+3], mem[a+2], mem[a+1], mem[a]});
    endfunction

    int errors = 0;
    int checks = 0;
    int tv_a [0:1023];
    int tv_b [0:1023];

    // int4 test vectors covering the whole [-8, 7] grid including both clips.
    task automatic gen_i4(input int n);
        for (int i = 0; i < n; i++) begin
            tv_a[i] = ((i * 5) % 16) - 8;
            tv_b[i] = ((i * 3) % 16) - 8;
            put4(A_ADDR, i, tv_a[i]);
            put4(B_ADDR, i, tv_b[i]);
        end
    endtask

    task automatic run_op(input logic [4:0] op, input int vlen,
                          input int m0, input int n);
        @(negedge clk);
        vpu_op      = op;
        vpu_src0    = A_ADDR;
        vpu_src1    = B_ADDR;
        vpu_dst     = D_ADDR;
        vpu_rq_word = {n[N_W-1:0], m0[M0_W-1:0]};
        vpu_vlen    = vlen[9:0];
        vpu_start   = 1'b1;
        @(negedge clk);
        vpu_start = 1'b0;
        wait (vpu_done);
        @(posedge clk);
    endtask

    task automatic expect4(input int idx, input int exp, input string tag);
        int got;
        got = get4(D_ADDR, idx);
        checks++;
        if (got !== exp) begin
            errors++;
            $display("  FAIL %s[%0d]: got %0d, expected %0d", tag, idx, got, exp);
        end
    endtask

    // The DUT's fixed point, in the testbench's own arithmetic.
    function automatic int narrow4(input int v, input int m0, input int n,
                                   input int lo);
        longint prod, round, shifted;
        prod    = longint'(v) * longint'(m0);
        round   = (n == 0) ? 0 : (longint'(1) << (n - 1));
        shifted = (prod + round) >>> n;
        if      (shifted >  7) return 7;
        else if (shifted < lo) return lo;
        else                   return int'(shifted);
    endfunction

    task automatic test_elem(input logic [4:0] op, input int vlen,
                             input int m0, input int n, input string tag);
        int raw, exp, lo;
        gen_i4(vlen);
        for (int i = 0; i < vlen; i++) put4(D_ADDR, i, 0);
        run_op(op, vlen, m0, n);
        lo = (op == VOP_DYT) ? -7 : -8;
        for (int i = 0; i < vlen; i++) begin
            case (op)
                VOP_ADD:  raw = tv_a[i] + tv_b[i];
                VOP_DYT:  raw = tv_a[i] + tv_b[i];
                VOP_RELU: raw = (tv_a[i] > 0) ? tv_a[i] : 0;
                default:  raw = tv_a[i];
            endcase
            exp = narrow4(raw, m0, n, lo);
            expect4(i, exp, tag);
        end
        $display("  %-18s vlen=%0d m0=%0d n=%0d", tag, vlen, m0, n);
    endtask

    task automatic test_dot(input int vlen, input string tag);
        int exp, got;
        gen_i4(vlen);
        run_op(VOP_DOT, vlen, 1, 0);
        exp = 0;
        for (int i = 0; i < vlen; i++) exp += tv_a[i] * tv_b[i];
        got = get32(D_ADDR);
        checks++;
        if (got !== exp) begin
            errors++;
            $display("  FAIL %s: got %0d, expected %0d", tag, got, exp);
        end
        $display("  %-18s vlen=%0d sum=%0d", tag, vlen, exp);
    endtask

    // DyT's only observable difference from ADD is the floor, so pin it.
    task automatic test_dyt_floor();
        gen_i4(LANES);
        put4(A_ADDR, 0, -8);
        put4(B_ADDR, 0, -8);
        run_op(VOP_DYT, LANES, 1, 0);
        expect4(0, -7, "DYT-floor");
        $display("  %-18s -8 + -8 -> %0d", "DYT-floor", get4(D_ADDR, 0));
    endtask

    // A destination byte is written whole, so the untouched half of a tail byte
    // takes nibble 0 rather than keeping what was there.
    task automatic test_tail_pack();
        gen_i4(LANES + 2);
        for (int i = 0; i < LANES + 4; i++) put4(D_ADDR, i, 7);
        run_op(VOP_REQUANT, LANES + 2, 1, 0);
        expect4(LANES + 0, tv_a[LANES + 0], "tail");
        expect4(LANES + 1, tv_a[LANES + 1], "tail");
        expect4(LANES + 2, 7, "tail-untouched");
        $display("  %-18s vlen=%0d", "tail-pack", LANES + 2);
    endtask

    // ---- ARGMAX -------------------------------------------------------------
    // src0 only: the int32 index of the largest element, ties to the lowest.

    task automatic set_a(input int idx, input int v);
        tv_a[idx] = v;
        put4(A_ADDR, idx, v);
    endtask

    task automatic fill_a(input int n, input int v);
        for (int i = 0; i < n; i++) set_a(i, v);
    endtask

    // Poison every element the op must not look at: the rest of the tail byte,
    // the next two words, and src1.
    task automatic poison_past(input int vlen);
        for (int i = vlen; i < vlen + 2*LANES; i++) begin
            put4(A_ADDR, i, 7);
            put4(B_ADDR, i, 7);
        end
        for (int i = 0; i < vlen; i++) put4(B_ADDR, i, 7);
    endtask

    task automatic test_argmax(input int vlen, input string tag);
        int exp, best, got;
        poison_past(vlen);
        for (int k = 0; k < 4; k++) mem[D_ADDR + k] = 8'hA5;
        run_op(VOP_ARGMAX, vlen, 1, 0);
        best = -9; exp = 0;
        for (int i = 0; i < vlen; i++)
            if (tv_a[i] > best) begin best = tv_a[i]; exp = i; end
        got = get32(D_ADDR);
        checks++;
        if (got !== exp) begin
            errors++;
            $display("  FAIL %s: got %0d, expected %0d (max=%0d)", tag, got, exp, best);
        end
        $display("  %-18s vlen=%0d idx=%0d max=%0d", tag, vlen, exp, best);
    endtask

    // A flat vector with one peak, to place the maximum in a chosen chunk.
    task automatic test_argmax_at(input int vlen, input int pos, input int peak,
                                  input int flat, input string tag);
        fill_a(vlen, flat);
        set_a(pos, peak);
        test_argmax(vlen, tag);
    endtask

    // Two ops back to back: the second's answer must not inherit the first's
    // running maximum.
    task automatic test_argmax_reset();
        test_argmax_at(40, 37, 7, -8, "ARGMAX-b2b-first");
        fill_a(40, -8);
        test_argmax(40, "ARGMAX-b2b-second");
    endtask

    initial begin
        vpu_start = 1'b0; vpu_op = '0;
        vpu_src0 = '0; vpu_src1 = '0; vpu_dst = '0;
        vpu_rq_word = '0; vpu_vlen = '0;
        for (int i = 0; i < MEM_SZ; i++) mem[i] = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== VPU testbench (int4) ====");

        // Elementwise: multi-chunk, exact boundary, and one lane pair.
        test_elem(VOP_ADD,     40, 1, 0, "ADD");
        test_elem(VOP_ADD,      8, 1, 0, "ADD-exact");
        test_elem(VOP_ADD,      2, 1, 0, "ADD-pair");
        test_elem(VOP_ADD,     24, 1, 1, "ADD-halve");
        test_elem(VOP_RELU,    40, 1, 0, "RELU");
        test_elem(VOP_RELU,    16, 3, 1, "RELU-rescale");
        test_elem(VOP_REQUANT, 40, 1, 0, "REQUANT-identity");
        test_elem(VOP_REQUANT, 32, 5, 2, "REQUANT-rescale");
        test_elem(VOP_REQUANT, 24, 4095, 0, "REQUANT-clip");
        test_elem(VOP_DYT,     40, 1, 0, "DYT");
        test_elem(VOP_DYT,     24, 1, 1, "DYT-halve");
        test_elem(VOP_DYT,     24, 4095, 0, "DYT-clip");
        test_dyt_floor();
        test_tail_pack();

        test_dot(40, "DOT");
        test_dot(24, "DOT-3chunk");
        test_dot(8,  "DOT-exact");

        // The ramp's maximum lands in the first chunk; the flat-plus-peak cases
        // walk it to the last chunk, the tail, and the tail's last lane.
        gen_i4(40);          test_argmax(40, "ARGMAX-ramp");
        gen_i4(LANES);       test_argmax(LANES, "ARGMAX-exact");
        gen_i4(2);           test_argmax(2, "ARGMAX-pair");
        test_argmax_at(40, 0,          7, -8, "ARGMAX-first");
        test_argmax_at(40, 39,         7, -8, "ARGMAX-last");
        test_argmax_at(40, LANES,      7, -8, "ARGMAX-chunk1");
        test_argmax_at(42, 41,         7, -8, "ARGMAX-tail-last");
        test_argmax_at(42, 40,         7, -8, "ARGMAX-tail-first");
        test_argmax_at(40, 25,        -1, -8, "ARGMAX-all-negative");
        test_argmax_at(40, 17,         0, -1, "ARGMAX-zero-peak");

        // Ties go to the lowest index, and an all-Q4_MIN vector still answers
        // 0 rather than letting the fold's padding win.
        fill_a(40, 3);  test_argmax(40, "ARGMAX-tie");
        fill_a(40, -8); test_argmax(40, "ARGMAX-all-min");
        fill_a(6,  -8); test_argmax(6,  "ARGMAX-all-min-tail");

        // A reduction writes no nibbles, so unlike every elementwise op it
        // takes an odd vlen — which is what the vocabulary is. The poison past
        // the end includes the other half of the last byte.
        gen_i4(13);      test_argmax(13, "ARGMAX-odd");
        test_argmax_at(13, 12, 7, -8, "ARGMAX-odd-last");
        test_argmax_at(11, 10, 7, -8, "ARGMAX-odd-tail3");
        test_argmax_at(9,   8, 7, -8, "ARGMAX-odd-tail1");
        fill_a(13, -8);  test_argmax(13, "ARGMAX-odd-all-min");

        test_argmax_reset();

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("VPU: ALL TESTS PASSED");
        else             $display("VPU: FAILED (%0d errors)", errors);
        $finish;
    end

    initial begin
        #200000;
        $display("VPU: TIMEOUT — DUT did not complete");
        $fatal(1);
    end

endmodule
