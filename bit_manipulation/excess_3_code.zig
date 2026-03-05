//! Excess-3 Code - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/excess_3_code.py

const std = @import("std");
const testing = std.testing;

fn writeNibble(out: []u8, value: u8) void {
    out[0] = if (((value >> 3) & 1) == 1) '1' else '0';
    out[1] = if (((value >> 2) & 1) == 1) '1' else '0';
    out[2] = if (((value >> 1) & 1) == 1) '1' else '0';
    out[3] = if ((value & 1) == 1) '1' else '0';
}

/// Converts a base-10 integer to excess-3 code string.
/// Negative values are clamped to zero to match Python behavior.
///
/// Time complexity: O(d), d = number of decimal digits
/// Space complexity: O(d)
pub fn excess3Code(allocator: std.mem.Allocator, number: i64) std.mem.Allocator.Error![]u8 {
    const normalized: i64 = if (number < 0) 0 else number;
    const decimal = try std.fmt.allocPrint(allocator, "{d}", .{normalized});
    defer allocator.free(decimal);

    const out = try allocator.alloc(u8, 2 + decimal.len * 4);
    out[0] = '0';
    out[1] = 'b';

    for (decimal, 0..) |ch, idx| {
        const digit: u8 = (ch - '0') + 3;
        writeNibble(out[2 + idx * 4 ..][0..4], digit);
    }
    return out;
}

test "excess-3 code: python examples" {
    const alloc = testing.allocator;

    const c0 = try excess3Code(alloc, 0);
    defer alloc.free(c0);
    try testing.expectEqualStrings("0b0011", c0);

    const c1 = try excess3Code(alloc, 3);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b0110", c1);

    const c2 = try excess3Code(alloc, 2);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b0101", c2);

    const c3 = try excess3Code(alloc, 20);
    defer alloc.free(c3);
    try testing.expectEqualStrings("0b01010011", c3);

    const c4 = try excess3Code(alloc, 120);
    defer alloc.free(c4);
    try testing.expectEqualStrings("0b010001010011", c4);
}

test "excess-3 code: boundary and extreme values" {
    const alloc = testing.allocator;

    const neg = try excess3Code(alloc, -123);
    defer alloc.free(neg);
    try testing.expectEqualStrings("0b0011", neg);

    const large = try excess3Code(alloc, std.math.maxInt(i64));
    defer alloc.free(large);
    try testing.expect(large.len > 2 + 18 * 4);
}
