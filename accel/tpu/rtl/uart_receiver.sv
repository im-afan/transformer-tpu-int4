// 115200 baud, 8N1
module uart_receiver #(
    parameter CLK_PER_BIT = 868 // 100 MHz, 115200 baud
) (
    input logic clk,
    input logic rst_n,

    input logic uart_rx, 
    output logic [7:0] data,
    output logic valid 
);

    localparam [3:0] IDLE = 0, START = 1, DATA = 2, STOP = 3;

    reg [3:0] state, state_n;
    reg rx_reg1, rx_reg2;

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            rx_reg1 <= 1;
            rx_reg2 <= 1;
        end else begin
            rx_reg1 <= uart_rx;
            rx_reg2 <= rx_reg1; 
        end
    end

    reg [15:0] clk_cnt;
    reg [3:0] bit_cnt;

    // Start-bit re-validation: see docs/uart_host.md. bit_cnt == 0 keeps this to
    // the start bit itself -- without the guard the glitch window would reopen
    // on top of data bit 0, since clk_cnt wraps every bit period.
    wire in_start_bit = (state == START) && (bit_cnt == 4'd0);
    wire start_glitch = in_start_bit && (clk_cnt <= CLK_PER_BIT / 2) &&  rx_reg2;
    wire start_ok     = in_start_bit && (clk_cnt == CLK_PER_BIT / 2) && !rx_reg2;

    always_comb begin
        state_n = state;
        case (state)
            IDLE: if(!rx_reg2) state_n = START;
            START: begin
                if(start_glitch)      state_n = IDLE;   // not a start bit after all
                else if(bit_cnt == 1) state_n = DATA;
            end
            DATA: if(bit_cnt == 9) state_n = STOP;
            STOP: if(rx_reg2) state_n = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state <= IDLE;
            clk_cnt <= 16'b0;
            bit_cnt <= 4'b0;
            valid <= 0;
            data <= 8'b0;
        end else begin
            state <= state_n;

            if(state == IDLE) begin
                clk_cnt <= 16'b0;
                bit_cnt <= 4'b0;
            end else begin
                if(clk_cnt + 1 == CLK_PER_BIT) begin
                    clk_cnt <= 16'b0;
                    bit_cnt <= bit_cnt + 1;
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end

            case (state) 
                IDLE: begin 
                     
                end
                START: begin
                    // Only once the start bit has held, so a rejected glitch
                    // cannot pull `valid` down under the consumer's edge detect.
                    if(start_ok) valid <= 0;
                end
                DATA: begin
                    if(clk_cnt == CLK_PER_BIT / 2) begin
                        // $display("data[%d] <= %d\n", bit_cnt-1, uart_rx);
                        data[bit_cnt-1] <= rx_reg2;
                    end
                end
                STOP: begin
                    valid <= 1;
                end
            endcase
        end
    end
endmodule