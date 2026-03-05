//! Single Bit Manipulation Operations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/single_bit_manipulation_operations.py

const std = @import("std");
const testing = std.testing;

pub const BitOperationError = error{PositionOutOfRange};

fn bitMask(position: usize) BitOperationError!u128 {
    if (position >= 128) return BitOperationError.PositionOutOfRange;
    return @as(u128, 1) << @intCast(position);
}

/// Sets bit at `position` to 1.
/// Time complexity: O(1)
pub fn setBit(number: u128, position: usize) BitOperationError!u128 {
    return number | try bitMask(position);
}

/// Clears bit at `position` to 0.
/// Time complexity: O(1)
pub fn clearBit(number: u128, position: usize) BitOperationError!u128 {
    return number & ~(try bitMask(position));
}

/// Flips bit at `position`.
/// Time complexity: O(1)
pub fn flipBit(number: u128, position: usize) BitOperationError!u128 {
    return number ^ (try bitMask(position));
}

/// Returns whether bit at `position` is set.
/// Time complexity: O(1)
pub fn isBitSet(number: u128, position: usize) BitOperationError!bool {
    const mask = try bitMask(position);
    return ((number & mask) >> @intCast(position)) == 1;
}

/// Returns bit value at `position` as `0` or `1`.
/// Time complexity: O(1)
pub fn getBit(number: u128, position: usize) BitOperationError!u1 {
    return if (try isBitSet(number, position)) 1 else 0;
}

test "single bit operations: python examples" {
    try testing.expectEqual(@as(u128, 15), try setBit(0b1101, 1));
    try testing.expectEqual(@as(u128, 32), try setBit(0b0, 5));
    try testing.expectEqual(@as(u128, 15), try setBit(0b1111, 1));

    try testing.expectEqual(@as(u128, 16), try clearBit(0b10010, 1));
    try testing.expectEqual(@as(u128, 0), try clearBit(0b0, 5));

    try testing.expectEqual(@as(u128, 7), try flipBit(0b101, 1));
    try testing.expectEqual(@as(u128, 4), try flipBit(0b101, 0));

    try testing.expectEqual(false, try isBitSet(0b1010, 0));
    try testing.expectEqual(true, try isBitSet(0b1010, 1));
    try testing.expectEqual(false, try isBitSet(0b1010, 2));
    try testing.expectEqual(true, try isBitSet(0b1010, 3));
    try testing.expectEqual(false, try isBitSet(0b0, 17));

    try testing.expectEqual(@as(u1, 0), try getBit(0b1010, 0));
    try testing.expectEqual(@as(u1, 1), try getBit(0b1010, 1));
    try testing.expectEqual(@as(u1, 0), try getBit(0b1010, 2));
    try testing.expectEqual(@as(u1, 1), try getBit(0b1010, 3));
}

test "single bit operations: boundary positions" {
    const msb: usize = 127;
    const value = try setBit(0, msb);
    try testing.expectEqual(@as(u128, 1) << 127, value);
    try testing.expect(try isBitSet(value, msb));
    try testing.expectEqual(@as(u128, 0), try clearBit(value, msb));
}

test "single bit operations: out of range position" {
    try testing.expectError(BitOperationError.PositionOutOfRange, setBit(0, 128));
    try testing.expectError(BitOperationError.PositionOutOfRange, clearBit(1, 128));
    try testing.expectError(BitOperationError.PositionOutOfRange, flipBit(1, 128));
    try testing.expectError(BitOperationError.PositionOutOfRange, isBitSet(1, 128));
    try testing.expectError(BitOperationError.PositionOutOfRange, getBit(1, 128));
}
