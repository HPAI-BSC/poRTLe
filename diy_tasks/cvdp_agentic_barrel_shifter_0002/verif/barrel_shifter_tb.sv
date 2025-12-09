`timescale 1ns / 1ps

module barrel_shifter_tb;

    // Testbench Signals
    reg [7:0] data_in;           // Input data
    reg [2:0] shift_bits;        // Number of bits to shift
    reg left_right;              // Direction of shift: 1 for left, 0 for right
    wire [7:0] data_out;         // Output data

    // Instantiate the DUT (Device Under Test)
    barrel_shifter uut (
        .data_in(data_in),
        .shift_bits(shift_bits),
        .left_right(left_right),
        .data_out(data_out)
    );

    // Task to calculate expected output for comparison
    function [7:0] expected_output(input [7:0] data_in, input [2:0] shift_bits, input left_right);
        begin
            if (left_right)  // Left shift
                expected_output = (data_in << shift_bits) & 8'hFF; // Mask to 8 bits
            else  // Right shift
                expected_output = (data_in >> shift_bits) & 8'hFF; // Mask to 8 bits
        end
    endfunction

    // Test procedure
    initial begin
        integer i;  // Loop variable
        reg [7:0] expected; // Holds the expected output

        // Display header
        $display("Starting Testbench for barrel_shifter_8bit...");
        $display("--------------------------------------------------");
        $display("|  Data_in  |  Shift | Left/Right |  Output  | Expected |");
        $display("--------------------------------------------------");

        // Apply test cases
        for (i = 0; i < 100; i = i + 1) begin
            // Generate random inputs
            data_in = $random % 256;  // 8-bit random value
            shift_bits = $random % 8; // 3-bit random value
            left_right = $random % 2; // Random left/right direction

            // Calculate expected output
            expected = expected_output(data_in, shift_bits, left_right);

            // Wait for a small delay to simulate propagation
            #5;

            // Display the results
            $display("| %b |   %0d   |     %0d     | %b | %b |",
                data_in, shift_bits, left_right, data_out, expected);

            // Check if the output matches the expected value
            if (data_out !== expected) begin
                $display("Test failed for data_in=%b, shift_bits=%0d, left_right=%0d. Expected=%b, Got=%b",
                    data_in, shift_bits, left_right, expected, data_out);
                $fatal; // Stop the simulation on failure
            end
        end

        // Test passed
        $display("All tests passed!");
        $finish;
    end
endmodule