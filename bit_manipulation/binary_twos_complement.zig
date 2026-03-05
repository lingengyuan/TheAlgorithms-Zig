//! Binary Two's Complement - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/binary_twos_complement.py

const std = @import("std");
const testing = std.testing;

pub const TwosComplementError = error{InputMustBeNegative};

fn bitLengthU128(value: u128) usize {
    return if (value == 0) 1 else 128 - @clz(value);
}

fn toBinaryNoPrefix(allocator: std.mem.Allocator, value: u128) std.mem.Allocator.Error![]u8 {
    const width = bitLengthU128(value);
    const out = try allocator.alloc(u8, width);
    for (0..width) |i| {
        const bit_index = width - 1 - i;
        const bit = (value >> @intCast(bit_index)) & 1;
        out[i] = if (bit == 1) '1' else '0';
    }
    return out;
}

/// Returns two's complement representation (`0b...`) for a non-positive integer.
/// Positive input is rejected to match Python reference behavior.
///
/// Time complexity: O(w)
/// Space complexity: O(w)
pub fn twosComplement(
    allocator: std.mem.Allocator,
    number: i64,
) (TwosComplementError || std.mem.Allocator.Error)![]u8 {
    if (number > 0) return TwosComplementError.InputMustBeNegative;
    if (number == 0) return allocator.dupe(u8, "0b0");

    const number_i128: i128 = number;
    const abs_number: u128 = @intCast(-number_i128);
    const binary_number_length = bitLengthU128(abs_number);

    const shift_value = @as(i128, 1) << @intCast(binary_number_length);
    const temp = @as(i128, @intCast(abs_number)) - shift_value; // negative
    const abs_temp: u128 = @intCast(-temp);
    const temp_bin = try toBinaryNoPrefix(allocator, abs_temp);
    defer allocator.free(temp_bin);

    const zero_count = binary_number_length - temp_bin.len;
    const body = try allocator.alloc(u8, 1 + zero_count + temp_bin.len);
    body[0] = '1';
    @memset(body[1 .. 1 + zero_count], '0');
    @memcpy(body[1 + zero_count ..], temp_bin);

    const out = try allocator.alloc(u8, 2 + body.len);
    out[0] = '0';
    out[1] = 'b';
    @memcpy(out[2..], body);
    allocator.free(body);
    return out;
}

test "binary twos complement: python examples" {
    const alloc = testing.allocator;

    const c0 = try twosComplement(alloc, 0);
    defer alloc.free(c0);
    try testing.expectEqualStrings("0b0", c0);

    const c1 = try twosComplement(alloc, -1);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b11", c1);

    const c2 = try twosComplement(alloc, -5);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b1011", c2);

    const c3 = try twosComplement(alloc, -17);
    defer alloc.free(c3);
    try testing.expectEqualStrings("0b101111", c3);

    const c4 = try twosComplement(alloc, -207);
    defer alloc.free(c4);
    try testing.expectEqualStrings("0b100110001", c4);
}

test "binary twos complement: positive rejected" {
    const alloc = testing.allocator;
    try testing.expectError(TwosComplementError.InputMustBeNegative, twosComplement(alloc, 1));
}

test "binary twos complement: extreme min value" {
    const alloc = testing.allocator;
    const c = try twosComplement(alloc, std.math.minInt(i64));
    defer alloc.free(c);
    try testing.expectEqual(@as(u8, '0'), c[0]);
    try testing.expectEqual(@as(u8, 'b'), c[1]);
    try testing.expect(c.len > 60);
}
