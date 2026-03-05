//! Binary Shifts - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/binary_shifts.py

const std = @import("std");
const testing = std.testing;

pub const ShiftError = error{NegativeValue};

fn bitLengthU64(value: u64) usize {
    return if (value == 0) 1 else 64 - @clz(value);
}

fn bitLengthU128(value: u128) usize {
    return if (value == 0) 1 else 128 - @clz(value);
}

fn formatBinaryNoPrefixFixedWidth(
    allocator: std.mem.Allocator,
    value: u128,
    width: usize,
) std.mem.Allocator.Error![]u8 {
    const out = try allocator.alloc(u8, width);
    for (0..width) |i| {
        const bit_index = width - 1 - i;
        const bit = (value >> @intCast(bit_index)) & 1;
        out[i] = if (bit == 1) '1' else '0';
    }
    return out;
}

/// Logical left shift rendered as binary string (`0b...`) with zero-appended bits.
///
/// Time complexity: O(w + s)
/// Space complexity: O(w + s)
pub fn logicalLeftShift(
    allocator: std.mem.Allocator,
    number: i64,
    shift_amount: i64,
) (ShiftError || std.mem.Allocator.Error)![]u8 {
    if (number < 0 or shift_amount < 0) return ShiftError.NegativeValue;

    const value: u64 = @intCast(number);
    const shift: usize = @intCast(shift_amount);
    const width = bitLengthU64(value);

    const out = try allocator.alloc(u8, 2 + width + shift);
    out[0] = '0';
    out[1] = 'b';

    for (0..width) |i| {
        const bit_index = width - 1 - i;
        const bit = (value >> @intCast(bit_index)) & 1;
        out[2 + i] = if (bit == 1) '1' else '0';
    }

    @memset(out[2 + width ..], '0');
    return out;
}

/// Logical right shift rendered as binary string (`0b...`) with width truncation.
///
/// Time complexity: O(w)
/// Space complexity: O(w)
pub fn logicalRightShift(
    allocator: std.mem.Allocator,
    number: i64,
    shift_amount: i64,
) (ShiftError || std.mem.Allocator.Error)![]u8 {
    if (number < 0 or shift_amount < 0) return ShiftError.NegativeValue;

    const value: u64 = @intCast(number);
    const shift: usize = @intCast(shift_amount);
    const width = bitLengthU64(value);

    if (shift >= width) return allocator.dupe(u8, "0b0");

    const out_width = width - shift;
    const out = try allocator.alloc(u8, 2 + out_width);
    out[0] = '0';
    out[1] = 'b';

    for (0..out_width) |i| {
        const bit_index = width - 1 - i;
        const bit = (value >> @intCast(bit_index)) & 1;
        out[2 + i] = if (bit == 1) '1' else '0';
    }
    return out;
}

/// Arithmetic right shift rendered as Python-style fixed-width sign-extended bit string.
///
/// Time complexity: O(w)
/// Space complexity: O(w)
pub fn arithmeticRightShift(
    allocator: std.mem.Allocator,
    number: i64,
    shift_amount: i64,
) (ShiftError || std.mem.Allocator.Error)![]u8 {
    if (shift_amount < 0) return ShiftError.NegativeValue;
    const shift: usize = @intCast(shift_amount);

    var binary_number: []u8 = undefined;
    if (number >= 0) {
        const positive: u128 = @intCast(number);
        const width = bitLengthU128(positive) + 1;
        binary_number = try formatBinaryNoPrefixFixedWidth(allocator, positive, width);
    } else {
        const number_i128: i128 = number;
        const abs_value: u128 = @intCast(-number_i128);
        const width = bitLengthU128(abs_value) + 1;
        const mask = (@as(u128, 1) << @intCast(width)) - 1;
        const two_complement = @as(u128, @bitCast(number_i128)) & mask;
        binary_number = try formatBinaryNoPrefixFixedWidth(allocator, two_complement, width);
    }
    defer allocator.free(binary_number);

    const out = try allocator.alloc(u8, 2 + binary_number.len);
    out[0] = '0';
    out[1] = 'b';
    const sign_bit = binary_number[0];

    if (shift >= binary_number.len) {
        @memset(out[2..], sign_bit);
        return out;
    }

    @memset(out[2 .. 2 + shift], sign_bit);
    @memcpy(out[2 + shift ..], binary_number[0 .. binary_number.len - shift]);
    return out;
}

test "binary shifts: logical left python examples" {
    const alloc = testing.allocator;

    const c1 = try logicalLeftShift(alloc, 0, 1);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b00", c1);

    const c2 = try logicalLeftShift(alloc, 1, 1);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b10", c2);

    const c3 = try logicalLeftShift(alloc, 1, 5);
    defer alloc.free(c3);
    try testing.expectEqualStrings("0b100000", c3);

    const c4 = try logicalLeftShift(alloc, 17, 2);
    defer alloc.free(c4);
    try testing.expectEqualStrings("0b1000100", c4);

    const c5 = try logicalLeftShift(alloc, 1983, 4);
    defer alloc.free(c5);
    try testing.expectEqualStrings("0b111101111110000", c5);
}

test "binary shifts: logical right python examples" {
    const alloc = testing.allocator;

    const c1 = try logicalRightShift(alloc, 0, 1);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b0", c1);

    const c2 = try logicalRightShift(alloc, 1, 1);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b0", c2);

    const c3 = try logicalRightShift(alloc, 1, 5);
    defer alloc.free(c3);
    try testing.expectEqualStrings("0b0", c3);

    const c4 = try logicalRightShift(alloc, 17, 2);
    defer alloc.free(c4);
    try testing.expectEqualStrings("0b100", c4);

    const c5 = try logicalRightShift(alloc, 1983, 4);
    defer alloc.free(c5);
    try testing.expectEqualStrings("0b1111011", c5);
}

test "binary shifts: arithmetic right python examples" {
    const alloc = testing.allocator;

    const c1 = try arithmeticRightShift(alloc, 0, 1);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b00", c1);

    const c2 = try arithmeticRightShift(alloc, 1, 1);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b00", c2);

    const c3 = try arithmeticRightShift(alloc, -1, 1);
    defer alloc.free(c3);
    try testing.expectEqualStrings("0b11", c3);

    const c4 = try arithmeticRightShift(alloc, 17, 2);
    defer alloc.free(c4);
    try testing.expectEqualStrings("0b000100", c4);

    const c5 = try arithmeticRightShift(alloc, -17, 2);
    defer alloc.free(c5);
    try testing.expectEqualStrings("0b111011", c5);

    const c6 = try arithmeticRightShift(alloc, -1983, 4);
    defer alloc.free(c6);
    try testing.expectEqualStrings("0b111110000100", c6);
}

test "binary shifts: validation and boundaries" {
    const alloc = testing.allocator;
    try testing.expectError(ShiftError.NegativeValue, logicalLeftShift(alloc, 1, -1));
    try testing.expectError(ShiftError.NegativeValue, logicalRightShift(alloc, -1, 1));
    try testing.expectError(ShiftError.NegativeValue, arithmeticRightShift(alloc, 1, -1));
}

test "binary shifts: extreme shifts" {
    const alloc = testing.allocator;

    const c1 = try logicalRightShift(alloc, 1, 100);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b0", c1);

    const c2 = try arithmeticRightShift(alloc, -1, 100);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b11", c2);

    const c3 = try arithmeticRightShift(alloc, std.math.minInt(i64), 1);
    defer alloc.free(c3);
    try testing.expect(c3.len > 3);
}
