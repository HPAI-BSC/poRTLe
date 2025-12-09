`timescale 1ns / 1ps

// Placeholder RTL module - replace with your design
module my_module (
    input wire clk,
    input wire reset,
    input wire [7:0] data_in,
    output reg [7:0] data_out,
    output reg valid
);

    // Simple passthrough logic (replace with your implementation)
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            data_out <= 8'b0;
            valid <= 1'b0;
        end else begin
            data_out <= data_in;
            valid <= 1'b1;
        end
    end

endmodule
