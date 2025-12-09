`timescale 1ns / 1ps

// Placeholder testbench - replace with your tests
module my_module_tb;

    reg clk;
    reg reset;
    reg [7:0] data_in;
    wire [7:0] data_out;
    wire valid;

    // Instantiate the module under test
    my_module uut (
        .clk(clk),
        .reset(reset),
        .data_in(data_in),
        .data_out(data_out),
        .valid(valid)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period
    end

    // Simple test sequence
    initial begin
        $dumpfile("my_module_tb.vcd");
        $dumpvars(0, my_module_tb);

        // Initialize
        reset = 1;
        data_in = 8'h00;
        #20;

        // Release reset
        reset = 0;
        #10;

        // Test: Simple passthrough
        $display("Test: Simple passthrough");
        data_in = 8'hAA;
        #20;
        if (data_out == 8'hAA && valid == 1)
            $display("PASS: Passthrough works");
        else
            $display("FAIL: Expected data_out=0xAA, valid=1, got data_out=0x%02X, valid=%b", data_out, valid);

        // Additional tests go here
        // Replace with your actual test cases

        $display("All tests completed");
        $finish;
    end

    // Monitor
    initial begin
        $monitor("Time=%0t reset=%b data_in=0x%02X data_out=0x%02X valid=%b",
                 $time, reset, data_in, data_out, valid);
    end

endmodule
