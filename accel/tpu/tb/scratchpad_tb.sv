`timescale 1ns/1ps
// Self-checking testbench for rtl/scratchpad.sv.
//
// Instantiates the scratchpad twice from one driver — MEM_STYLE="BRAM" and
// "REG" — so the storage knob stays a pure synthesis choice, and checks both
// against a byte reference model.
//
// Coverage: every port's word/byte roundtrip, cross-port coherence, the write
// strobes, concurrent access to different banks, the priority chain when two
// requesters name the same bank, and read-first on one bank.
//
// Run:  make TEST=scratchpad sim

module scratchpad_tb;

    localparam int N          = 8;
    localparam int ADDR_W     = 12;
    localparam int BANK_WORDS = 64;
    localparam int S_BYTES    = 4;

    localparam int WORD_BITS  = N * 4;
    localparam int WORD_BYTES = N / 2;
    localparam int NBANK      = (1 << ADDR_W) / (WORD_BYTES * BANK_WORDS);
    localparam int BANK_BYTES = BANK_WORDS * WORD_BYTES;
    localparam int MEM_SZ     = 1 << ADDR_W;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---- shared bus ---------------------------------------------------------
    logic                  A_re;   logic [ADDR_W-1:0] A_addr;
    logic                  B_re;   logic [ADDR_W-1:0] B_addr;
    logic                  C_en, C_we; logic [ADDR_W-1:0] C_addr;
    logic [WORD_BITS-1:0]  C_wdata;
    logic                  V_re;   logic [ADDR_W-1:0] V_raddr;
    logic                  V_we;   logic [ADDR_W-1:0] V_waddr;
    logic [WORD_BITS-1:0]  V_wdata; logic [WORD_BYTES-1:0] V_wstrb;
    logic                  dma_re; logic [ADDR_W-1:0] dma_raddr;
    logic                  dma_we; logic [ADDR_W-1:0] dma_waddr;
    logic [7:0]            dma_wdata;
    logic                  s_re, s_we; logic [ADDR_W-1:0] s_addr;
    logic [S_BYTES*8-1:0]  s_wdata;

    logic [WORD_BITS-1:0] A_rdata_b, A_rdata_r, B_rdata_b, B_rdata_r;
    logic [WORD_BITS-1:0] C_rdata_b, C_rdata_r, V_rdata_b, V_rdata_r;
    logic [7:0]           dma_rdata_b, dma_rdata_r;
    logic [S_BYTES*8-1:0] s_rdata_b, s_rdata_r;
    logic A_gnt_b, B_gnt_b, C_gnt_b, V_rgnt_b, V_wgnt_b;
    logic dma_rgnt_b, dma_wgnt_b, s_rgnt_b, s_wgnt_b;
    logic A_gnt_r, B_gnt_r, C_gnt_r, V_rgnt_r, V_wgnt_r;
    logic dma_rgnt_r, dma_wgnt_r, s_rgnt_r, s_wgnt_r;

    scratchpad #(
        .MEM_STYLE("BRAM"), .N(N), .ADDR_W(ADDR_W),
        .BANK_WORDS(BANK_WORDS), .S_BYTES(S_BYTES)
    ) dut_b (
        .clk(clk), .rst_n(rst_n),
        .A_re(A_re), .A_addr(A_addr), .A_rdata(A_rdata_b), .A_gnt(A_gnt_b),
        .B_re(B_re), .B_addr(B_addr), .B_rdata(B_rdata_b), .B_gnt(B_gnt_b),
        .C_en(C_en), .C_we(C_we), .C_addr(C_addr), .C_wdata(C_wdata),
        .C_rdata(C_rdata_b), .C_gnt(C_gnt_b),
        .V_re(V_re), .V_raddr(V_raddr), .V_rdata(V_rdata_b), .V_rgnt(V_rgnt_b),
        .V_we(V_we), .V_waddr(V_waddr), .V_wdata(V_wdata), .V_wstrb(V_wstrb),
        .V_wgnt(V_wgnt_b),
        .dma_re(dma_re), .dma_raddr(dma_raddr), .dma_rdata(dma_rdata_b),
        .dma_rgnt(dma_rgnt_b),
        .dma_we(dma_we), .dma_waddr(dma_waddr), .dma_wdata(dma_wdata),
        .dma_wgnt(dma_wgnt_b),
        .s_re(s_re), .s_we(s_we), .s_addr(s_addr), .s_wdata(s_wdata),
        .s_rdata(s_rdata_b), .s_rgnt(s_rgnt_b), .s_wgnt(s_wgnt_b)
    );

    scratchpad #(
        .MEM_STYLE("REG"), .N(N), .ADDR_W(ADDR_W),
        .BANK_WORDS(BANK_WORDS), .S_BYTES(S_BYTES)
    ) dut_r (
        .clk(clk), .rst_n(rst_n),
        .A_re(A_re), .A_addr(A_addr), .A_rdata(A_rdata_r), .A_gnt(A_gnt_r),
        .B_re(B_re), .B_addr(B_addr), .B_rdata(B_rdata_r), .B_gnt(B_gnt_r),
        .C_en(C_en), .C_we(C_we), .C_addr(C_addr), .C_wdata(C_wdata),
        .C_rdata(C_rdata_r), .C_gnt(C_gnt_r),
        .V_re(V_re), .V_raddr(V_raddr), .V_rdata(V_rdata_r), .V_rgnt(V_rgnt_r),
        .V_we(V_we), .V_waddr(V_waddr), .V_wdata(V_wdata), .V_wstrb(V_wstrb),
        .V_wgnt(V_wgnt_r),
        .dma_re(dma_re), .dma_raddr(dma_raddr), .dma_rdata(dma_rdata_r),
        .dma_rgnt(dma_rgnt_r),
        .dma_we(dma_we), .dma_waddr(dma_waddr), .dma_wdata(dma_wdata),
        .dma_wgnt(dma_wgnt_r),
        .s_re(s_re), .s_we(s_we), .s_addr(s_addr), .s_wdata(s_wdata),
        .s_rdata(s_rdata_r), .s_rgnt(s_rgnt_r), .s_wgnt(s_wgnt_r)
    );

    // ---- reference ----------------------------------------------------------
    logic [7:0] ref_mem [0:MEM_SZ-1];
    int errors = 0, checks = 0;

    function automatic [WORD_BITS-1:0] ref_word(input [ADDR_W-1:0] a);
        logic [ADDR_W-1:0] wa;
        wa = a & ~ADDR_W'(WORD_BYTES - 1);
        ref_word = '0;
        for (int k = 0; k < WORD_BYTES; k++) ref_word[k*8 +: 8] = ref_mem[wa + k[ADDR_W-1:0]];
    endfunction

    task automatic chk(input logic cond, input string tag);
        checks++;
        if (!cond) begin
            errors++;
            $display("  FAIL %s (t=%0t)", tag, $time);
        end
    endtask

    task automatic chk_word(input [WORD_BITS-1:0] gb, input [WORD_BITS-1:0] gr,
                            input [ADDR_W-1:0] a, input string tag);
        logic [WORD_BITS-1:0] e;
        e = ref_word(a);
        checks++;
        if (gb !== e || gr !== e) begin
            errors++;
            $display("  FAIL %-16s @0x%03h: BRAM %h REG %h exp %h", tag, a, gb, gr, e);
        end
    endtask

    task automatic idle_bus();
        A_re=0; B_re=0; C_en=0; C_we=0; V_re=0; V_we=0;
        dma_re=0; dma_we=0; s_re=0; s_we=0;
    endtask

    // ---- port drivers -------------------------------------------------------
    task automatic wr_c(input [ADDR_W-1:0] a, input [WORD_BITS-1:0] d);
        @(negedge clk); idle_bus(); C_en=1; C_we=1; C_addr=a; C_wdata=d;
        @(negedge clk); idle_bus();
        for (int k = 0; k < WORD_BYTES; k++) ref_mem[a + k[ADDR_W-1:0]] = d[k*8 +: 8];
    endtask

    task automatic wr_v(input [ADDR_W-1:0] a, input [WORD_BITS-1:0] d,
                        input [WORD_BYTES-1:0] strb);
        @(negedge clk); idle_bus(); V_we=1; V_waddr=a; V_wdata=d; V_wstrb=strb;
        @(negedge clk); idle_bus();
        for (int k = 0; k < WORD_BYTES; k++)
            if (strb[k]) ref_mem[a + k[ADDR_W-1:0]] = d[k*8 +: 8];
    endtask

    task automatic wr_s(input [ADDR_W-1:0] a, input [S_BYTES*8-1:0] d);
        @(negedge clk); idle_bus(); s_we=1; s_addr=a; s_wdata=d;
        @(negedge clk); idle_bus();
        for (int k = 0; k < S_BYTES; k++) ref_mem[a + k[ADDR_W-1:0]] = d[k*8 +: 8];
    endtask

    task automatic wr_dma(input [ADDR_W-1:0] a, input [7:0] d);
        @(negedge clk); idle_bus(); dma_we=1; dma_waddr=a; dma_wdata=d;
        @(negedge clk); idle_bus();
        ref_mem[a] = d;
    endtask

    task automatic rd_a(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); idle_bus(); A_re=1; A_addr=a;
        @(negedge clk); idle_bus();
        chk_word(A_rdata_b, A_rdata_r, a, tag);
    endtask

    task automatic rd_b(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); idle_bus(); B_re=1; B_addr=a;
        @(negedge clk); idle_bus();
        chk_word(B_rdata_b, B_rdata_r, a, tag);
    endtask

    task automatic rd_c(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); idle_bus(); C_en=1; C_we=0; C_addr=a;
        @(negedge clk); idle_bus();
        chk_word(C_rdata_b, C_rdata_r, a, tag);
    endtask

    task automatic rd_v(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); idle_bus(); V_re=1; V_raddr=a;
        @(negedge clk); idle_bus();
        chk_word(V_rdata_b, V_rdata_r, a, tag);
    endtask

    task automatic rd_s(input [ADDR_W-1:0] a, input string tag);
        logic [S_BYTES*8-1:0] e;
        @(negedge clk); idle_bus(); s_re=1; s_addr=a;
        @(negedge clk); idle_bus();
        e = '0;
        for (int k = 0; k < S_BYTES; k++) e[k*8 +: 8] = ref_mem[a + k[ADDR_W-1:0]];
        checks++;
        if (s_rdata_b !== e || s_rdata_r !== e) begin
            errors++;
            $display("  FAIL %-16s @0x%03h: BRAM %h REG %h exp %h",
                     tag, a, s_rdata_b, s_rdata_r, e);
        end
    endtask

    task automatic rd_dma(input [ADDR_W-1:0] a, input string tag);
        @(negedge clk); idle_bus(); dma_re=1; dma_raddr=a;
        @(negedge clk); idle_bus();
        checks++;
        if (dma_rdata_b !== ref_mem[a] || dma_rdata_r !== ref_mem[a]) begin
            errors++;
            $display("  FAIL %-16s @0x%03h: BRAM %02h REG %02h exp %02h",
                     tag, a, dma_rdata_b, dma_rdata_r, ref_mem[a]);
        end
    endtask

    // ---- stimulus -----------------------------------------------------------
    logic [ADDR_W-1:0] p0, p1;

    initial begin
        idle_bus();
        A_addr=0; B_addr=0; C_addr=0; C_wdata=0;
        V_raddr=0; V_waddr=0; V_wdata=0; V_wstrb=0;
        dma_raddr=0; dma_waddr=0; dma_wdata=0; s_addr=0; s_wdata=0;
        for (int i = 0; i < MEM_SZ; i++) ref_mem[i] = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("==== Scratchpad testbench (%0d banks x %0d words x %0d B) ====",
                 NBANK, BANK_WORDS, WORD_BYTES);

        // --- per-port roundtrips ---------------------------------------------
        wr_c(12'h010, 32'hDEADBEEF);
        rd_c(12'h010, "C-roundtrip");
        rd_a(12'h010, "C->A");
        rd_b(12'h010, "C->B");
        rd_v(12'h010, "C->V");

        wr_v(12'h020, 32'h01234567, {WORD_BYTES{1'b1}});
        rd_v(12'h020, "V-full");
        wr_v(12'h020, 32'hAABBCCDD, 4'b0101);
        rd_v(12'h020, "V-strobe");

        wr_s(12'h030, 32'h89ABCDEF);
        rd_s(12'h030, "S-roundtrip");
        rd_a(12'h030, "S->A");

        // --- DMA byte lanes: every offset within one word ---------------------
        for (int off = 0; off < WORD_BYTES; off++) begin
            wr_dma(12'h040 + off[ADDR_W-1:0], 8'hA0 + off[7:0]);
            rd_dma(12'h040 + off[ADDR_W-1:0], "DMA-byte");
        end
        rd_a(12'h040, "DMA->A");

        // --- one bank per requester: everything proceeds together -------------
        // Bank b covers bytes [b*BANK_BYTES, (b+1)*BANK_BYTES).
        wr_c(ADDR_W'(0*BANK_BYTES), 32'h11111111);
        wr_c(ADDR_W'(1*BANK_BYTES), 32'h22222222);
        wr_c(ADDR_W'(2*BANK_BYTES), 32'h33333333);
        wr_c(ADDR_W'(3*BANK_BYTES), 32'h44444444);

        @(negedge clk); idle_bus();
        A_re=1; A_addr=ADDR_W'(0*BANK_BYTES);
        B_re=1; B_addr=ADDR_W'(1*BANK_BYTES);
        V_re=1; V_raddr=ADDR_W'(2*BANK_BYTES);
        dma_re=1; dma_raddr=ADDR_W'(3*BANK_BYTES);
        @(negedge clk);
        chk(A_gnt_b && B_gnt_b && V_rgnt_b && dma_rgnt_b, "4-bank grants");
        idle_bus();
        chk_word(A_rdata_b, A_rdata_r, ADDR_W'(0*BANK_BYTES), "4-bank A");
        chk_word(B_rdata_b, B_rdata_r, ADDR_W'(1*BANK_BYTES), "4-bank B");
        chk_word(V_rdata_b, V_rdata_r, ADDR_W'(2*BANK_BYTES), "4-bank V");
        chk(dma_rdata_b === ref_mem[3*BANK_BYTES], "4-bank DMA");

        // --- same bank: the priority chain decides ---------------------------
        @(negedge clk); idle_bus();
        A_re=1; A_addr=ADDR_W'(0*BANK_BYTES);
        B_re=1; B_addr=ADDR_W'(0*BANK_BYTES + WORD_BYTES);
        @(negedge clk);
        chk(A_gnt_b && !B_gnt_b, "A beats B");
        idle_bus();
        chk_word(A_rdata_b, A_rdata_r, ADDR_W'(0*BANK_BYTES), "A won the bank");

        @(negedge clk); idle_bus();
        V_re=1; V_raddr=ADDR_W'(1*BANK_BYTES);
        dma_re=1; dma_raddr=ADDR_W'(1*BANK_BYTES);
        s_re=1;  s_addr=ADDR_W'(1*BANK_BYTES);
        @(negedge clk);
        chk(V_rgnt_b && !dma_rgnt_b && !s_rgnt_b, "V beats DMA beats S");
        idle_bus();

        @(negedge clk); idle_bus();
        C_en=1; C_we=1; C_addr=ADDR_W'(2*BANK_BYTES); C_wdata=32'hFACEFACE;
        V_we=1; V_waddr=ADDR_W'(2*BANK_BYTES); V_wdata=32'h00000000;
        V_wstrb={WORD_BYTES{1'b1}};
        dma_we=1; dma_waddr=ADDR_W'(2*BANK_BYTES); dma_wdata=8'hFF;
        @(negedge clk); idle_bus();
        chk(1'b1, "write chain issued");
        for (int k = 0; k < WORD_BYTES; k++)
            ref_mem[2*BANK_BYTES + k] = 32'hFACEFACE >> (k*8);
        rd_c(ADDR_W'(2*BANK_BYTES), "C beats V and DMA");

        // --- a read and a write to the same bank share it ---------------------
        p0 = ADDR_W'(4*BANK_BYTES);
        wr_c(p0, 32'h12345678);
        @(negedge clk); idle_bus();
        V_re=1; V_raddr=p0;                       // old value
        C_en=1; C_we=1; C_addr=p0; C_wdata=32'h99999999;
        @(negedge clk); idle_bus();
        chk(V_rgnt_b && C_gnt_b, "read+write same bank");
        chk(V_rdata_b === 32'h12345678 && V_rdata_r === 32'h12345678, "read-first");
        for (int k = 0; k < WORD_BYTES; k++)
            ref_mem[p0 + k[ADDR_W-1:0]] = 32'h99999999 >> (k*8);
        rd_v(p0, "read-first-post");

        // --- a sweep across every bank ----------------------------------------
        for (int bnk = 0; bnk < NBANK; bnk++) begin
            p1 = ADDR_W'(bnk*BANK_BYTES + WORD_BYTES);
            wr_v(p1, 32'h0F0F0000 + bnk[31:0], {WORD_BYTES{1'b1}});
        end
        for (int bnk = 0; bnk < NBANK; bnk++) begin
            p1 = ADDR_W'(bnk*BANK_BYTES + WORD_BYTES);
            rd_a(p1, "bank-sweep");
        end

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("SCRATCHPAD: ALL TESTS PASSED");
        else             $display("SCRATCHPAD: FAILED (%0d errors)", errors);
        $finish;
    end

    initial begin
        #500000;
        $display("SCRATCHPAD: TIMEOUT — testbench did not complete");
        $fatal(1);
    end

endmodule
