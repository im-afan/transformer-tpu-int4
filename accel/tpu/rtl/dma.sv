`timescale 1ns/1ps
// DMA engine. Drives the external SRAM pins itself and moves `rows` rows of
// `len` int4 elements between DRAM and the scratchpad, one byte per clock on a
// fill and one per two clocks on a spill. Also serves the UART host's single
// byte accesses while no command is running. See docs/dma.md.

module dma #(
    parameter int MEM_ADDR_W = 19,   // external SRAM byte address
    parameter int ADDR_W     = 16    // scratchpad byte address
) (
    input  logic clk,
    input  logic rst_n,

    // ---- external SRAM pins -------------------------------------------------
    output logic [MEM_ADDR_W-1:0] sram_addr,
    inout  wire  [7:0]            sram_data,
    output logic                  sram_we,
    output logic                  sram_ce,
    output logic                  sram_oen,

    // ---- dispatch -----------------------------------------------------------
    input  logic                  dma_start,
    input  logic                  dma_op,           // 0: DRAM->spad, 1: spad->DRAM
    input  logic [15:0]           dma_len,          // int4 elements per row
    input  logic [15:0]           dma_rows,
    input  logic [15:0]           dma_dram_stride,  // bytes between DRAM rows
    input  logic [15:0]           dma_spad_stride,  // bytes between scratchpad rows
    input  logic [MEM_ADDR_W-1:0] dma_dram_base,
    input  logic [ADDR_W-1:0]     dma_spad_base,
    output logic                  dma_busy,
    output logic                  dma_done,

    // ---- scratchpad byte port -----------------------------------------------
    output logic              spad_re,
    output logic [ADDR_W-1:0] spad_raddr,
    input  logic [7:0]        spad_rdata,
    input  logic              spad_rgnt,
    output logic              spad_we,
    output logic [ADDR_W-1:0] spad_waddr,
    output logic [7:0]        spad_wdata,
    input  logic              spad_wgnt,

    // ---- UART host byte port (serviced while idle) --------------------------
    input  logic                  host_start,
    input  logic                  host_we,
    input  logic [MEM_ADDR_W-1:0] host_addr,
    input  logic [7:0]            host_din,
    output logic [7:0]            host_dout,
    output logic                  host_busy,
    output logic                  host_done
);

    typedef enum logic [2:0] {
        S_IDLE,
        S_FILL,      // DRAM -> scratchpad, one byte per clock
        S_SPILL,     // scratchpad -> DRAM, one byte per two-clock write beat
        S_HOST_RD,
        S_HOST_WR,
        S_DONE
    } state_t;
    state_t state;

    // Zero stride means densely packed rows.
    wire [15:0] row_bytes = (dma_len + 16'd1) >> 1;
    wire [15:0] dram_str  = (dma_dram_stride != 16'd0) ? dma_dram_stride : row_bytes;
    wire [15:0] spad_str  = (dma_spad_stride != 16'd0) ? dma_spad_stride : row_bytes;

    // Fetch side (the fill's whole walk, or the spill's scratchpad reads).
    logic [15:0]            f_col, f_row;
    logic [ADDR_W-1:0]      spad_row_ptr;
    // Write side (the spill's DRAM writes; it trails the fetch by one byte).
    logic [15:0]            w_col, w_row;
    logic [MEM_ADDR_W-1:0]  dram_row_ptr;

    logic [MEM_ADDR_W-1:0]  addr_q;
    logic [7:0]             din_q, hold;
    logic                   drive_en, we_win, beat;
    logic                   pending, fetch_inflight, wr_active;

    assign sram_data = drive_en ? din_q : 8'bz;
    assign sram_addr = addr_q;

    wire fetch_more = (f_row < dma_rows);
    wire write_more = (w_row < dma_rows);
    wire row_last_f = (f_col + 16'd1 >= row_bytes);
    wire row_last_w = (w_col + 16'd1 >= row_bytes);
    // A beat's pins are registered on the last clock of the previous beat, so
    // back-to-back bytes sustain one write every two clocks.
    wire beat_start = (state == S_SPILL) && pending && (!wr_active || beat);

    assign spad_raddr = spad_row_ptr + ADDR_W'(f_col);
    assign spad_waddr = spad_row_ptr + ADDR_W'(f_col);
    assign spad_wdata = sram_data;
    assign spad_we    = (state == S_FILL) && fetch_more;
    assign spad_re    = (state == S_SPILL) && fetch_more && !fetch_inflight
                        && (!pending || beat_start);

    assign dma_busy  = (state == S_FILL) || (state == S_SPILL);
    assign dma_done  = (state == S_DONE);
    assign host_busy = (state == S_HOST_RD) || (state == S_HOST_WR);

    // WE# is driven from the falling edge so both of its edges sit half a clock
    // clear of every address and data change.
    always_ff @(negedge clk or negedge rst_n) begin
        if (!rst_n) sram_we <= 1'b1;
        else        sram_we <= ~we_win;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            f_col          <= '0;
            f_row          <= '0;
            w_col          <= '0;
            w_row          <= '0;
            spad_row_ptr   <= '0;
            dram_row_ptr   <= '0;
            addr_q         <= '0;
            din_q          <= '0;
            hold           <= '0;
            drive_en       <= 1'b0;
            we_win         <= 1'b0;
            beat           <= 1'b0;
            pending        <= 1'b0;
            fetch_inflight <= 1'b0;
            wr_active      <= 1'b0;
            sram_ce        <= 1'b1;
            sram_oen       <= 1'b1;
            host_dout      <= '0;
            host_done      <= 1'b0;
        end else begin
            host_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    sram_ce        <= 1'b1;
                    sram_oen       <= 1'b1;
                    drive_en       <= 1'b0;
                    we_win         <= 1'b0;
                    pending        <= 1'b0;
                    fetch_inflight <= 1'b0;
                    wr_active      <= 1'b0;
                    beat           <= 1'b0;
                    f_col          <= '0;
                    f_row          <= '0;
                    w_col          <= '0;
                    w_row          <= '0;

                    if (dma_start) begin
                        spad_row_ptr <= dma_spad_base;
                        dram_row_ptr <= dma_dram_base;
                        addr_q       <= dma_dram_base;
                        if (dma_rows == 16'd0 || row_bytes == 16'd0) begin
                            state <= S_DONE;
                        end else begin
                            sram_ce <= 1'b0;
                            if (dma_op) begin
                                drive_en <= 1'b1;
                                state    <= S_SPILL;
                            end else begin
                                sram_oen <= 1'b0;
                                state    <= S_FILL;
                            end
                        end
                    end else if (host_start) begin
                        addr_q  <= host_addr;
                        sram_ce <= 1'b0;
                        if (host_we) begin
                            din_q    <= host_din;
                            drive_en <= 1'b1;
                            we_win   <= 1'b1;
                            beat     <= 1'b0;
                            state    <= S_HOST_WR;
                        end else begin
                            sram_oen <= 1'b0;
                            state    <= S_HOST_RD;
                        end
                    end
                end

                // ---- fill: the SRAM holds the byte until the write lands -----
                S_FILL: if (spad_wgnt) begin
                    if (row_last_f) begin
                        f_col        <= '0;
                        f_row        <= f_row + 16'd1;
                        spad_row_ptr <= spad_row_ptr + ADDR_W'(spad_str);
                        dram_row_ptr <= dram_row_ptr + MEM_ADDR_W'(dram_str);
                        addr_q       <= dram_row_ptr + MEM_ADDR_W'(dram_str);
                        if (f_row + 16'd1 >= dma_rows) begin
                            sram_ce  <= 1'b1;
                            sram_oen <= 1'b1;
                            state    <= S_DONE;
                        end
                    end else begin
                        f_col  <= f_col + 16'd1;
                        addr_q <= addr_q + MEM_ADDR_W'(1);
                    end
                end

                // ---- spill: fetch one byte ahead of the write beat -----------
                S_SPILL: begin
                    if (spad_re && spad_rgnt) fetch_inflight <= 1'b1;

                    if (fetch_inflight) begin
                        fetch_inflight <= 1'b0;
                        hold           <= spad_rdata;
                        pending        <= 1'b1;
                        if (row_last_f) begin
                            f_col        <= '0;
                            f_row        <= f_row + 16'd1;
                            spad_row_ptr <= spad_row_ptr + ADDR_W'(spad_str);
                        end else begin
                            f_col <= f_col + 16'd1;
                        end
                    end

                    if (beat_start) begin
                        addr_q    <= dram_row_ptr + MEM_ADDR_W'(w_col);
                        din_q     <= hold;
                        we_win    <= 1'b1;
                        wr_active <= 1'b1;
                        beat      <= 1'b0;
                        pending   <= 1'b0;
                        if (row_last_w) begin
                            w_col        <= '0;
                            w_row        <= w_row + 16'd1;
                            dram_row_ptr <= dram_row_ptr + MEM_ADDR_W'(dram_str);
                        end else begin
                            w_col <= w_col + 16'd1;
                        end
                    end else if (wr_active) begin
                        beat <= 1'b1;
                        if (beat) begin
                            // WE# rose on this clock's negedge; the pins may move.
                            wr_active <= 1'b0;
                            if (!write_more) begin
                                sram_ce  <= 1'b1;
                                drive_en <= 1'b0;
                                state    <= S_DONE;
                            end
                        end else begin
                            we_win <= 1'b0;
                        end
                    end
                end

                S_HOST_RD: begin
                    host_dout <= sram_data;
                    host_done <= 1'b1;
                    sram_ce   <= 1'b1;
                    sram_oen  <= 1'b1;
                    state     <= S_IDLE;
                end

                S_HOST_WR: begin
                    beat <= 1'b1;
                    if (beat) begin
                        host_done <= 1'b1;
                        sram_ce   <= 1'b1;
                        drive_en  <= 1'b0;
                        state     <= S_IDLE;
                    end else begin
                        we_win <= 1'b0;
                    end
                end

                S_DONE: state <= S_IDLE;

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
