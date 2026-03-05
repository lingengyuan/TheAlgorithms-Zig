//! Binary Coded Decimal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/binary_coded_decimal.py

const std = @import("std");
const testing = std.testing;

fn writeNibble(out: []u8, value: u8) void {
    out[0] = if (((value >> 3) & 1) == 1) '1' else '0';
    out[1] = if (((value >> 2) & 1) == 1) '1' else '0';
    out[2] = if (((value >> 1) & 1) == 1) '1' else '0';
    out[3] = if ((value & 1) == 1) '1' else '0';
}

/// Converts a base-10 integer to binary-coded decimal string.
/// Negative values are clamped to zero to match Python behavior.
///
/// Time complexity: O(d), d = number of decimal digits
/// Space complexity: O(d)
pub fn binaryCodedDecimal(allocator: std.mem.Allocator, number: i64) std.mem.Allocator.Error![]u8 {
    const normalized: i64 = if (number < 0) 0 else number;
    const decimal = try std.fmt.allocPrint(allocator, "{d}", .{normalized});
    defer allocator.free(decimal);

    const out = try allocator.alloc(u8, 2 + decimal.len * 4);
    out[0] = '0';
    out[1] = 'b';

    for (decimal, 0..) |ch, idx| {
        const digit = ch - '0';
        writeNibble(out[2 + idx * 4 ..][0..4], digit);
    }
    return out;
}

test "binary coded decimal: python examples" {
    const alloc = testing.allocator;

    const c1 = try binaryCodedDecimal(alloc, -2);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b0000", c1);

    const c2 = try binaryCodedDecimal(alloc, -1);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b0000", c2);

    const c3 = try binaryCodedDecimal(alloc, 0);
    defer alloc.free(c3);
    try testing.expectEqualStrings("0b0000", c3);

    const c4 = try binaryCodedDecimal(alloc, 3);
    defer alloc.free(c4);
    try testing.expectEqualStrings("0b0011", c4);

    const c5 = try binaryCodedDecimal(alloc, 2);
    defer alloc.free(c5);
    try testing.expectEqualStrings("0b0010", c5);

    const c6 = try binaryCodedDecimal(alloc, 12);
    defer alloc.free(c6);
    try testing.expectEqualStrings("0b00010010", c6);

    const c7 = try binaryCodedDecimal(alloc, 987);
    defer alloc.free(c7);
    try testing.expectEqualStrings("0b100110000111", c7);
}

test "binary coded decimal: extreme large number" {
    const alloc = testing.allocator;
    const out = try binaryCodedDecimal(alloc, std.math.maxInt(i64));
    defer alloc.free(out);
    try testing.expect(out.len > 2 + 18 * 4);
    try testing.expectEqual(@as(u8, '0'), out[0]);
    try testing.expectEqual(@as(u8, 'b'), out[1]);
}
