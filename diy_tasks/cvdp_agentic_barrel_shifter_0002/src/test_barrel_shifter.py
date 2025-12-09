import cocotb
import os
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, ClockCycles, Timer, Join
import random

def verify(data_in, shift_bits, mode, left_right, data_width=8):
    """Helper function to calculate expected output."""
    if mode == 0b00:  # Logical Shift
        if left_right == 1:
            expected = (data_in << shift_bits) & ((1 << data_width) - 1)
        else:
            expected = (data_in >> shift_bits)
    elif mode == 0b01:  # Arithmetic Shift
        if left_right == 1:
            expected = (data_in << shift_bits) & ((1 << data_width) - 1)
        else:
            # Sign-extend the MSB for arithmetic shift
            sign_bit = data_in >> (data_width - 1) & 1
            expected = (data_in >> shift_bits) | (
                ((1 << shift_bits) - 1) << (data_width - shift_bits) if sign_bit else 0
            )

    return expected


# ----------------------------------------
# - Tests
# ----------------------------------------

@cocotb.test()
async def logic_shift(dut):

    for i in range(10):
        data_in = int((1.0 - random.random()) * 2 ** 8)
        shift_bits = random.randint(0, 7)
        mode = 0
        left_right = random.randint(0, 1)

        dut.data_in.value = data_in
        dut.shift_bits.value = shift_bits
        dut.left_right.value = left_right
        dut.shift_mode.value = mode

        await Timer(10, units="ns")

        expected = verify(data_in, shift_bits, mode, left_right)

        if dut.data_out.value == expected:
            print("Logic shift process is successful")
        assert dut.data_out.value == expected, f"Computed and DUT outputs of barrel shifter are not correct {data_in}, {shift_bits}, {mode}, {left_right}, {expected}, {dut.data_out.value}"

@cocotb.test()
async def arithmatic_shift(dut):
      
    for i in range(10):
        data_in = int((1.0 - random.random()) * 2 ** 8)
        shift_bits = random.randint(0, 7)
        mode = 1
        left_right = random.randint(0, 1)

        dut.data_in.value = data_in
        dut.shift_bits.value = shift_bits
        dut.left_right.value = left_right
        dut.shift_mode.value = mode

        await Timer(10, units="ns")

        expected = verify(data_in, shift_bits, mode, left_right)

        if dut.data_out.value == expected:
            print("Arithmatic shift process is successful")
        assert dut.data_out.value == expected, f"Computed and DUT outputs of barrel shifter are not correct {bin(data_in)}, {shift_bits}, {mode}, {left_right}, {expected}, {dut.data_out.value}"
        await Timer(5, units="ns")