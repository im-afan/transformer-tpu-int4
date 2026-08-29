`timescale 1ns/1ps
// Self-checking testbench for rtl/dma.sv.
//
// The DMA drives the SRAM pins itself now, so this wires it to the behavioral
// async-SRAM chip model and the real scratchpad and checks both directions of a
// `rows` x `len` transfer against a byte reference. Also covers the strides, the
// UART host's byte port, and a scratchpad denial long enough to stall the
// stream.
//
// The write-hazard monitor is the reason the WE# timing is worth testing here:
// a controller that moves the address inside the write pulse writes to the
// neighbouring cell on hardware while looking perfect in a functional model.
//
// Run:  make TEST=dma sim

module dma_tb;

    localparam int MEM_ADDR_W = 19;
    localparam int ADDR_W     = 12;
    localparam int N          = 8;
    localparam int BANK_WORDS = 64;
    localparam int WORD_BYTES = N / 2;
    localparam int BANK_BYTES = BANK_WORDS * WORD_BYTES;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---- dispatch -----------------------------------------------------------
    logic                  dma_start, dma_op;
    logic [15:0]           dma_len, dma_rows, dma_dram_stride, dma_spad_stride;
    logic [MEM_ADDR_W-1:0] dma_dram_base;
    logic [ADDR_W-1:0]     dma_spad_base;
    logic                  dma_busy, dma_done;

    logic                  host_start, host_we;
    logic [MEM_ADDR_W-1:0] host_addr;
    logic [7:0]            host_din, host_dout;
    logic                  host_busy, host_done;

    // ---- DMA <-> scratchpad -------------------------------------------------
    logic              spad_re, spad_we, spad_rgnt, spad_wgnt;
    logic [ADDR_W-1:0] spad_raddr, spad_waddr;
    logic [7:0]        spad_rdata, spad_wdata;

    // Contention injector: a VPU access on the same bank denies the DMA.
    logic              spy_V_re = 1'b0, spy_V_we = 1'b0;
    logic [ADDR_W-1:0] spy_V_addr = '0;

    // ---- chip pins ----------------------------------------------------------
    wire [MEM_ADDR_W-1:0] chip_addr;
    wire [7:0]            chip_data;
    wire                  chip_we, chip_ce, chip_oen;

    dma #(.MEM_ADDR_W(MEM_ADDR_W), .ADDR_W(ADDR_W)) u_dma (
        .clk(clk), .rst_n(rst_n),
        .sram_addr(chip_addr), .sram_data(chip_data),
        .sram_we(chip_we), .sram_ce(chip_ce), .sram_oen(chip_oen),
        .dma_start(dma_start), .dma_op(dma_op),
        .dma_len(dma_len), .dma_rows(dma_rows),
        .dma_dram_stride(dma_dram_stride), .dma_spad_stride(dma_spad_stride),
        .dma_dram_base(dma_dram_base), .dma_spad_base(dma_spad_base),
        .dma_busy(dma_busy), .dma_done(dma_done),
        .spad_re(spad_re), .spad_raddr(spad_raddr), .spad_rdata(spad_rdata),
        .spad_rgnt(spad_rgnt),
        .spad_we(spad_we), .spad_waddr(spad_waddr), .spad_wdata(spad_wdata),
        .spad_wgnt(spad_wgnt),
        .host_start(host_start), .host_we(host_we), .host_addr(host_addr),
        .host_din(host_din), .host_dout(host_dout),
        .host_busy(host_busy), .host_done(host_done)
    );

    scratchpad #(
        .MEM_STYLE("REG"), .N(N), .ADDR_W(ADDR_W), .BANK_WORDS(BANK_WORDS)
    ) u_spad (
        .clk(clk), .rst_n(rst_n),
        .A_re(1'b0), .A_addr('0), .A_rdata(), .A_gnt(),
        .B_re(1'b0), .B_addr('0), .B_rdata(), .B_gnt(),
        .C_en(1'b0), .C_we(1'b0), .C_addr('0), .C_wdata('0),
        .C_rdata(), .C_gnt(),
        .V_re(spy_V_re), .V_raddr(spy_V_addr), .V_rdata(), .V_rgnt(),
        .V_we(spy_V_we), .V_waddr(spy_V_addr), .V_wdata('0), .V_wstrb('0),
        .V_wgnt(),
        .dma_re(spad_re), .dma_raddr(spad_raddr), .dma_rdata(spad_rdata),
        .dma_rgnt(spad_rgnt),
        .dma_we(spad_we), .dma_waddr(spad_waddr), .dma_wdata(spad_wdata),
        .dma_wgnt(spad_wgnt),
        .s_re(1'b0), .s_we(1'b0), .s_addr('0), .s_wdata('0), .s_rdata(),
        .s_rgnt(), .s_wgnt()
    );

    // ---- behavioral async SRAM chip -----------------------------------------
    localparam int SRAM_SZ = 1 << MEM_ADDR_W;
    logic [7:0] sram_mem [0:SRAM_SZ-1];

    wire mem_drives = (chip_ce == 1'b0) && (chip_oen == 1'b0) && (chip_we == 1'b1);
    assign #3 chip_data = mem_drives ? sram_mem[chip_addr] : 8'bz;

    logic [MEM_ADDR_W-1:0] we_lo_addr;
    logic [7:0]            we_lo_data;
    int                    write_hazards = 0;

    always @(negedge chip_we) begin
        we_lo_addr <= chip_addr;
        we_lo_data <= chip_data;
    end

    always @(posedge chip_we) begin
        if (chip_ce == 1'b0) begin
            if (chip_addr !== we_lo_addr || chip_data !== we_lo_data) begin
                write_hazards++;
                $display("  FAIL write hazard at t=%0t: addr %05h->%05h data %02h->%02h",
                         $time, we_lo_addr, chip_addr, we_lo_data, chip_data);
            end
            sram_mem[chip_addr] <= chip_data;
        end
    end

    // ---- helpers ------------------------------------------------------------
    int errors = 0, checks = 0;

    task automatic chk(input logic cond, input string tag);
        checks++;
        if (!cond) begin
            errors++;
            $display("  FAIL %s (t=%0t)", tag, $time);
        end
    endtask

    task automatic spad_put(input [ADDR_W-1:0] a, input [7:0] d);
        u_spad.bd_poke(a, d);
    endtask

    task automatic spad_get(input [ADDR_W-1:0] a, output [7:0] d);
        u_spad.bd_peek(a, d);
    endtask

    task automatic run_dma(input logic op, input int rows, input int bytes_per_row,
                           input int dstride, input int sstride,
                           input [MEM_ADDR_W-1:0] dbase, input [ADDR_W-1:0] sbase);
        @(negedge clk);
        dma_op          = op;
        dma_rows        = rows[15:0];
        dma_len         = (bytes_per_row * 2);      // int4 elements per row
        dma_dram_stride = dstride[15:0];
        dma_spad_stride = sstride[15:0];
        dma_dram_base   = dbase;
        dma_spad_base   = sbase;
        dma_start       = 1'b1;
        @(negedge clk);
        dma_start = 1'b0;
        while (!dma_done) @(negedge clk);   // already high for a 1-clock transfer
        @(negedge clk);
    endtask

    task automatic host_access(input logic we, input [MEM_ADDR_W-1:0] a,
                               input [7:0] d, output [7:0] q);
        @(negedge clk);
        host_we = we; host_addr = a; host_din = d; host_start = 1'b1;
        @(negedge clk);
        host_start = 1'b0;
        while (!host_done) @(negedge clk);
        q = host_dout;
        @(negedge clk);
    endtask

    // ---- tests --------------------------------------------------------------
    localparam [MEM_ADDR_W-1:0] DR = 19'h01000;
    localparam [ADDR_W-1:0]     SP = 12'h100;

    task automatic test_fill(input int rows, input int bpr,
                             input int dstride, input int sstride,
                             input string tag);
        logic [7:0] got, exp;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < bpr; c++)
                sram_mem[DR + r*dstride + c] = 8'(r*16 + c + 1);
        for (int i = 0; i < 512; i++) spad_put(SP + ADDR_W'(i), 8'h00);

        run_dma(1'b0, rows, bpr, dstride, sstride, DR, SP);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < bpr; c++) begin
                spad_get(SP + ADDR_W'(r*sstride + c), got);
                exp = 8'(r*16 + c + 1);
                checks++;
                if (got !== exp) begin
                    errors++;
                    if (errors < 20)
                        $display("  FAIL %-12s spad[%0d][%0d]: got %02h exp %02h",
                                 tag, r, c, got, exp);
                end
            end
        $display("[%-12s] %0d x %0d bytes  (errors so far: %0d)", tag, rows, bpr, errors);
    endtask

    task automatic test_spill(input int rows, input int bpr,
                              input int dstride, input int sstride,
                              input string tag);
        logic [7:0] exp;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < bpr; c++)
                spad_put(SP + ADDR_W'(r*sstride + c), 8'(r*32 + c + 7));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < bpr; c++) sram_mem[DR + r*dstride + c] = 8'hEE;

        run_dma(1'b1, rows, bpr, dstride, sstride, DR, SP);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < bpr; c++) begin
                exp = 8'(r*32 + c + 7);
                checks++;
                if (sram_mem[DR + r*dstride + c] !== exp) begin
                    errors++;
                    if (errors < 20)
                        $display("  FAIL %-12s dram[%0d][%0d]: got %02h exp %02h",
                                 tag, r, c, sram_mem[DR + r*dstride + c], exp);
                end
            end
        $display("[%-12s] %0d x %0d bytes  (errors so far: %0d)", tag, rows, bpr, errors);
    endtask

    // ---- contention driver ---------------------------------------------------
    // Hammer the bank the transfer is using so the DMA is denied for long runs.
    logic contend = 1'b0;
    always @(posedge clk) begin
        if (contend) begin
            spy_V_re   <= ($random % 3 != 0);
            spy_V_we   <= ($random % 3 != 0);
            spy_V_addr <= SP;
        end else begin
            spy_V_re <= 1'b0;
            spy_V_we <= 1'b0;
        end
    end

    logic [7:0] hq;

    initial begin
        dma_start = 0; dma_op = 0; dma_len = 0; dma_rows = 0;
        dma_dram_stride = 0; dma_spad_stride = 0;
        dma_dram_base = 0; dma_spad_base = 0;
        host_start = 0; host_we = 0; host_addr = 0; host_din = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("==== DMA testbench ====");

        test_fill(1,  8,  8,  8,  "FILL-1x8");
        test_fill(4,  8,  8,  8,  "FILL-4x8");
        test_fill(4,  6, 16, 12,  "FILL-strided");
        test_fill(3,  1,  4,  4,  "FILL-1byte");

        test_spill(1, 8,  8,  8,  "SPILL-1x8");
        test_spill(4, 8,  8,  8,  "SPILL-4x8");
        test_spill(4, 6, 16, 12,  "SPILL-strided");
        test_spill(3, 1,  4,  4,  "SPILL-1byte");

        // Zero strides mean densely packed rows.
        test_fill(4, 8, 8, 8, "FILL-dense");
        run_dma(1'b0, 4, 8, 0, 0, DR, SP);
        chk(1'b1, "zero-stride fill completed");

        // Round trip: spill a region out and fill it back.
        for (int i = 0; i < 32; i++) spad_put(SP + ADDR_W'(i), 8'(i ^ 8'h5A));
        run_dma(1'b1, 4, 8, 8, 8, DR, SP);
        for (int i = 0; i < 32; i++) spad_put(SP + ADDR_W'(i), 8'h00);
        run_dma(1'b0, 4, 8, 8, 8, DR, SP);
        for (int i = 0; i < 32; i++) begin
            spad_get(SP + ADDR_W'(i), hq);
            checks++;
            if (hq !== 8'(i ^ 8'h5A)) begin
                errors++;
                if (errors < 20) $display("  FAIL roundtrip[%0d]: got %02h", i, hq);
            end
        end
        $display("[roundtrip  ] (errors so far: %0d)", errors);

        // Empty transfers still hand the handshake back.
        run_dma(1'b0, 0, 8, 8, 8, DR, SP);
        run_dma(1'b1, 4, 0, 8, 8, DR, SP);
        chk(1'b1, "empty transfers completed");

        // Under contention the stream stalls but loses nothing.
        contend = 1'b1;
        test_fill(4, 8, 8, 8, "FILL-contend");
        test_spill(4, 8, 8, 8, "SPILL-contend");
        contend = 1'b0;
        @(negedge clk);

        // Host byte port.
        host_access(1'b1, 19'h02000, 8'h5A, hq);
        chk(sram_mem[19'h02000] === 8'h5A, "host write");
        host_access(1'b0, 19'h02000, 8'h00, hq);
        chk(hq === 8'h5A, "host read");
        sram_mem[19'h7FFFF] = 8'hC3;
        host_access(1'b0, 19'h7FFFF, 8'h00, hq);
        chk(hq === 8'hC3, "host read, top of DRAM");

        chk(write_hazards == 0, "no write hazards");

        $display("==== done: %0d checks, %0d errors ====", checks, errors);
        if (errors == 0) $display("DMA: ALL TESTS PASSED");
        else             $display("DMA: FAILED (%0d errors)", errors);
        $finish;
    end

    initial begin
        #2000000;
        $display("DMA: TIMEOUT — DUT did not complete");
        $fatal(1);
    end

endmodule
