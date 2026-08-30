`timescale 1ns/1ps
// Per-run event counters for the TPU core. See docs/perf_counters.md.

module perf_counters #(
    parameter int N = 6,        // number of counters
    parameter int W = 32        // width of each counter
) (
    input  logic           clk,
    input  logic           rst_n,
    input  logic           run,      // scalar_unit.busy — defines the run window
    input  logic [N-1:0]   ev,       // ev[i] high on a clock counter i should count
    output logic [N*W-1:0] counts    // counter i at counts[i*W +: W]
);

    logic run_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_q  <= 1'b0;
            counts <= '0;
        end else begin
            run_q <= run;
            for (int i = 0; i < N; i++) begin
                if (run && !run_q) begin
                    // Run starts: restart, counting this clock if already active.
                    counts[i*W +: W] <= ev[i] ? W'(1) : W'(0);
                end else if (run && ev[i] && !(&counts[i*W +: W])) begin
                    counts[i*W +: W] <= counts[i*W +: W] + W'(1);
                end
                // else: idle, event inactive, or saturated — hold.
            end
        end
    end

endmodule
