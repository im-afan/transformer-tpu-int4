`timescale 1ns/1ps
// Banked working memory. NBANK independent BRAMs, each one N*4-bit word wide,
// mapped as contiguous address regions. One access is one bank word; two
// requesters proceed together whenever they name different banks.
// See docs/scratchpad.md.

module scratchpad #(
    parameter     MEM_STYLE  = "BRAM",   // "BRAM" -> block RAM, "REG" -> flip-flops
    parameter int N          = 8,        // systolic array size; word = N*4 bits
    parameter int ADDR_W     = 16,       // byte address width
    parameter int BANK_WORDS = 1024,     // words per bank (RAMB36 at 32 bits wide)
    parameter int S_BYTES    = 4,        // CPU word port width in bytes
    parameter     INIT_FILE  = ""        // optional $readmemh preload, one byte/line
) (
    input  logic clk,
    input  logic rst_n,

    // MXU A operand (read)
    input  logic              A_re,
    input  logic [ADDR_W-1:0] A_addr,
    output logic [N*4-1:0]    A_rdata,
    output logic              A_gnt,

    // MXU B operand (read)
    input  logic              B_re,
    input  logic [ADDR_W-1:0] B_addr,
    output logic [N*4-1:0]    B_rdata,
    output logic              B_gnt,

    // MXU C result (read or write, one address)
    input  logic              C_en,
    input  logic              C_we,
    input  logic [ADDR_W-1:0] C_addr,
    input  logic [N*4-1:0]    C_wdata,
    output logic [N*4-1:0]    C_rdata,
    output logic              C_gnt,

    // VPU
    input  logic              V_re,
    input  logic [ADDR_W-1:0] V_raddr,
    output logic [N*4-1:0]    V_rdata,
    output logic              V_rgnt,
    input  logic              V_we,
    input  logic [ADDR_W-1:0] V_waddr,
    input  logic [N*4-1:0]    V_wdata,
    input  logic [N/2-1:0]    V_wstrb,
    output logic              V_wgnt,

    // DMA, one byte per access
    input  logic              dma_re,
    input  logic [ADDR_W-1:0] dma_raddr,
    output logic [7:0]        dma_rdata,
    output logic              dma_rgnt,
    input  logic              dma_we,
    input  logic [ADDR_W-1:0] dma_waddr,
    input  logic [7:0]        dma_wdata,
    output logic              dma_wgnt,

    // CPU window, one aligned word per access
    input  logic                 s_re,
    input  logic                 s_we,
    input  logic [ADDR_W-1:0]    s_addr,
    input  logic [S_BYTES*8-1:0] s_wdata,
    output logic [S_BYTES*8-1:0] s_rdata,
    output logic                 s_rgnt,
    output logic                 s_wgnt
);

    localparam int WORD_BITS  = N * 4;
    localparam int WORD_BYTES = N / 2;
    localparam int WOFF_W     = $clog2(WORD_BYTES);          // byte within a word
    localparam int BANK_AW    = $clog2(BANK_WORDS);          // word within a bank
    localparam int NBANK      = (1 << ADDR_W) / (WORD_BYTES * BANK_WORDS);
    localparam int S_BITS     = S_BYTES * 8;

    initial begin
        if (WORD_BYTES < S_BYTES)
            $fatal(1, "scratchpad: word (%0d B) narrower than the CPU port (%0d B)",
                   WORD_BYTES, S_BYTES);
        if (NBANK * BANK_WORDS * WORD_BYTES != (1 << ADDR_W))
            $fatal(1, "scratchpad: %0d banks x %0d words x %0d B != 2**%0d",
                   NBANK, BANK_WORDS, WORD_BYTES, ADDR_W);
        if ((NBANK & (NBANK - 1)) != 0)
            $fatal(1, "scratchpad: NBANK (%0d) is not a power of two", NBANK);
    end

    function automatic int bank_of(input logic [ADDR_W-1:0] a);
        bank_of = int'(a >> (WOFF_W + BANK_AW));
    endfunction

    function automatic logic [BANK_AW-1:0] row_of(input logic [ADDR_W-1:0] a);
        row_of = BANK_AW'(a >> WOFF_W);
    endfunction

    // -------------------------------------------------------------------------
    // Requesters, highest priority first: MXU (A > B > C), VPU, DMA, CPU.
    // -------------------------------------------------------------------------
    localparam int NRD = 6;   // A, B, C, V, DMA, S
    localparam int NWR = 4;   // C, V, DMA, S

    logic [NRD-1:0]    rd_req, rd_gnt;
    logic [ADDR_W-1:0] rd_addr [0:NRD-1];
    logic [NWR-1:0]    wr_req, wr_gnt;
    logic [ADDR_W-1:0] wr_addr [0:NWR-1];

    logic [WORD_BITS-1:0]  wr_word [0:NWR-1];
    logic [WORD_BYTES-1:0] wr_lane [0:NWR-1];

    wire [WOFF_W-1:0] dma_wlane = dma_waddr[WOFF_W-1:0];
    wire [WOFF_W-1:0] s_wlane   = s_addr[WOFF_W-1:0];

    always_comb begin
        rd_req[0] = A_re;              rd_addr[0] = A_addr;
        rd_req[1] = B_re;              rd_addr[1] = B_addr;
        rd_req[2] = C_en && !C_we;     rd_addr[2] = C_addr;
        rd_req[3] = V_re;              rd_addr[3] = V_raddr;
        rd_req[4] = dma_re;            rd_addr[4] = dma_raddr;
        rd_req[5] = s_re;              rd_addr[5] = s_addr;

        wr_req[0] = C_en && C_we;      wr_addr[0] = C_addr;
        wr_req[1] = V_we;              wr_addr[1] = V_waddr;
        wr_req[2] = dma_we;            wr_addr[2] = dma_waddr;
        wr_req[3] = s_we;              wr_addr[3] = s_addr;

        wr_word[0] = C_wdata;
        wr_lane[0] = {WORD_BYTES{1'b1}};
        wr_word[1] = V_wdata;
        wr_lane[1] = V_wstrb;
        wr_word[2] = {WORD_BYTES{dma_wdata}};
        wr_lane[2] = WORD_BYTES'(1) << dma_wlane;
        wr_word[3] = WORD_BITS'(s_wdata) << (int'(s_wlane) * 8);
        wr_lane[3] = WORD_BYTES'({S_BYTES{1'b1}}) << s_wlane;
    end

    // -------------------------------------------------------------------------
    // Per-bank arbitration. Lowest requester index wins its bank; every other
    // bank is free to serve someone else on the same clock.
    // -------------------------------------------------------------------------
    integer           rd_bank [0:NRD-1];
    integer           wr_bank [0:NWR-1];
    integer           rd_sel  [0:NBANK-1];
    integer           wr_sel  [0:NBANK-1];
    logic [NBANK-1:0] rd_any, wr_any;

    always_comb begin
        for (int r = 0; r < NRD; r++) rd_bank[r] = bank_of(rd_addr[r]);
        for (int w = 0; w < NWR; w++) wr_bank[w] = bank_of(wr_addr[w]);

        for (int b = 0; b < NBANK; b++) begin
            rd_sel[b] = 0;
            rd_any[b] = 1'b0;
            for (int r = NRD-1; r >= 0; r--)
                if (rd_req[r] && rd_bank[r] == b) begin
                    rd_sel[b] = r;
                    rd_any[b] = 1'b1;
                end
            wr_sel[b] = 0;
            wr_any[b] = 1'b0;
            for (int w = NWR-1; w >= 0; w--)
                if (wr_req[w] && wr_bank[w] == b) begin
                    wr_sel[b] = w;
                    wr_any[b] = 1'b1;
                end
        end

        for (int r = 0; r < NRD; r++) rd_gnt[r] = rd_req[r] && (rd_sel[rd_bank[r]] == r);
        for (int w = 0; w < NWR; w++) wr_gnt[w] = wr_req[w] && (wr_sel[wr_bank[w]] == w);
    end

    assign A_gnt    = rd_gnt[0];
    assign B_gnt    = rd_gnt[1];
    assign C_gnt    = C_we ? wr_gnt[0] : rd_gnt[2];
    assign V_rgnt   = rd_gnt[3];
    assign dma_rgnt = rd_gnt[4];
    assign s_rgnt   = rd_gnt[5];
    assign V_wgnt   = wr_gnt[1];
    assign dma_wgnt = wr_gnt[2];
    assign s_wgnt   = wr_gnt[3];

    // -------------------------------------------------------------------------
    // Bank storage. One write port and one read port each, read-first.
    // -------------------------------------------------------------------------
    logic                  bank_rd_en   [0:NBANK-1];
    logic [BANK_AW-1:0]    bank_rd_row  [0:NBANK-1];
    logic                  bank_wr_en   [0:NBANK-1];
    logic [BANK_AW-1:0]    bank_wr_row  [0:NBANK-1];
    logic [WORD_BITS-1:0]  bank_wr_data [0:NBANK-1];
    logic [WORD_BYTES-1:0] bank_wr_strb [0:NBANK-1];
    logic [WORD_BITS-1:0]  bank_dout    [0:NBANK-1];

    always_comb begin
        for (int b = 0; b < NBANK; b++) begin
            bank_rd_en[b]   = rd_any[b];
            bank_rd_row[b]  = row_of(rd_addr[rd_sel[b]]);
            bank_wr_en[b]   = wr_any[b];
            bank_wr_row[b]  = row_of(wr_addr[wr_sel[b]]);
            bank_wr_data[b] = wr_word[wr_sel[b]];
            bank_wr_strb[b] = wr_any[b] ? wr_lane[wr_sel[b]] : '0;
        end
    end

// synthesis translate_off
    logic [BANK_AW-1:0]   bd_row;
    logic [ADDR_W-1:0]    bd_bank;
    logic [WOFF_W-1:0]    bd_lane;
    logic [7:0]           bd_wdata;
    logic [WORD_BITS-1:0] bd_rd [0:NBANK-1];
    event                 bd_wr_ev;
// synthesis translate_on

    genvar b;
    generate
        for (b = 0; b < NBANK; b++) begin : g_bank
            if (MEM_STYLE == "REG" || MEM_STYLE == "registers") begin : g_style
                (* ram_style = "registers" *) logic [WORD_BITS-1:0] mem [0:BANK_WORDS-1];

                always_ff @(posedge clk) begin
                    for (int k = 0; k < WORD_BYTES; k++)
                        if (bank_wr_en[b] && bank_wr_strb[b][k])
                            mem[bank_wr_row[b]][k*8 +: 8] <= bank_wr_data[b][k*8 +: 8];
                    if (bank_rd_en[b]) bank_dout[b] <= mem[bank_rd_row[b]];
                end

// synthesis translate_off
                assign bd_rd[b] = mem[bd_row];
                always @(bd_wr_ev)
                    if (bd_bank == ADDR_W'(b)) mem[bd_row][int'(bd_lane)*8 +: 8] = bd_wdata;
// synthesis translate_on

                if (INIT_FILE != "") begin : g_preload
                    logic [7:0] img [0:(1<<ADDR_W)-1];
                    initial begin
                        $readmemh(INIT_FILE, img);
                        for (int r = 0; r < BANK_WORDS; r++)
                            for (int k = 0; k < WORD_BYTES; k++)
                                mem[r][k*8 +: 8] =
                                    img[(b*BANK_WORDS + r)*WORD_BYTES + k];
                    end
                end
            end else begin : g_style
                (* ram_style = "block" *) logic [WORD_BITS-1:0] mem [0:BANK_WORDS-1];

                always_ff @(posedge clk) begin
                    for (int k = 0; k < WORD_BYTES; k++)
                        if (bank_wr_en[b] && bank_wr_strb[b][k])
                            mem[bank_wr_row[b]][k*8 +: 8] <= bank_wr_data[b][k*8 +: 8];
                    if (bank_rd_en[b]) bank_dout[b] <= mem[bank_rd_row[b]];
                end

// synthesis translate_off
                assign bd_rd[b] = mem[bd_row];
                always @(bd_wr_ev)
                    if (bd_bank == ADDR_W'(b)) mem[bd_row][int'(bd_lane)*8 +: 8] = bd_wdata;
// synthesis translate_on

                if (INIT_FILE != "") begin : g_preload
                    logic [7:0] img [0:(1<<ADDR_W)-1];
                    initial begin
                        $readmemh(INIT_FILE, img);
                        for (int r = 0; r < BANK_WORDS; r++)
                            for (int k = 0; k < WORD_BYTES; k++)
                                mem[r][k*8 +: 8] =
                                    img[(b*BANK_WORDS + r)*WORD_BYTES + k];
                    end
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Read return. bank_dout is the port register, so data is valid exactly one
    // cycle after a granted request; the bank select and the sub-word offset are
    // registered alongside it.
    // -------------------------------------------------------------------------
    integer           rd_bank_q [0:NRD-1];
    logic [WOFF_W-1:0] dma_roff_q, s_roff_q;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int r = 0; r < NRD; r++) rd_bank_q[r] <= 0;
            dma_roff_q <= '0;
            s_roff_q   <= '0;
        end else begin
            for (int r = 0; r < NRD; r++) if (rd_gnt[r]) rd_bank_q[r] <= rd_bank[r];
            if (rd_gnt[4]) dma_roff_q <= dma_raddr[WOFF_W-1:0];
            if (rd_gnt[5]) s_roff_q   <= s_addr[WOFF_W-1:0];
        end
    end

    assign A_rdata   = bank_dout[rd_bank_q[0]];
    assign B_rdata   = bank_dout[rd_bank_q[1]];
    assign C_rdata   = bank_dout[rd_bank_q[2]];
    assign V_rdata   = bank_dout[rd_bank_q[3]];
    assign dma_rdata = 8'(bank_dout[rd_bank_q[4]] >> (int'(dma_roff_q) * 8));
    assign s_rdata   = S_BITS'(bank_dout[rd_bank_q[5]] >> (int'(s_roff_q) * 8));

    // -------------------------------------------------------------------------
    // Simulation backdoor: one byte at an absolute address, bypassing the ports.
    // -------------------------------------------------------------------------
// synthesis translate_off
`ifndef SYNTHESIS
    task automatic bd_peek(input logic [ADDR_W-1:0] a, output logic [7:0] d);
        logic [WORD_BITS-1:0] w;
        bd_row  = row_of(a);
        #0;
        w = bd_rd[bank_of(a)];
        d = w[int'(a[WOFF_W-1:0])*8 +: 8];
    endtask

    task automatic bd_poke(input logic [ADDR_W-1:0] a, input logic [7:0] d);
        bd_bank  = ADDR_W'(bank_of(a));
        bd_row   = row_of(a);
        bd_lane  = a[WOFF_W-1:0];
        bd_wdata = d;
        ->bd_wr_ev;
        #0;
    endtask

    // The MXU cannot stall its operand feed, so A and B sharing a bank is a
    // software error rather than something the arbiter can fix.
    bit warn_ab = 1;
    always @(posedge clk)
        if (rst_n && warn_ab && A_re && B_re && bank_of(A_addr) == bank_of(B_addr)) begin
            $display("[%0t] scratchpad: MXU A and B in bank %0d — B denied",
                     $time, bank_of(A_addr));
            warn_ab = 0;
        end
`endif
// synthesis translate_on

endmodule
